"use client";

/**
 * In-browser inference with TensorFlow.js. The model (MobileNetV1 fine-tuned on field photos,
 * docs/field.md; float16 weights, 6.55 MB) takes raw RGB pixels in [0, 255] at 224 x 224;
 * preprocessing is inside the model. Images never leave the device.
 *
 * Verified against the Keras model in Node (docs/figures/phase6/tfjs_verify*.json): same top-1
 * class on 100/100 images (max probability difference 0.012), also with the 4-view test-time
 * augmentation (0.008).
 */

import type * as TF from "@tensorflow/tfjs";

import { type ModelInfo, gradcam } from "@/lib/gradcam";
import { predictTta } from "@/lib/tta";

export const MODEL_URL = "/model/model.json";
export const MODEL_INFO_URL = "/model/leafdx.json";
export const INPUT_SIZE = 224;

type Loaded = { tf: typeof TF; model: TF.GraphModel; backend: string; info: ModelInfo };

let loading: Promise<Loaded> | null = null;

async function pickBackend(tf: typeof TF): Promise<string> {
  const wasm = await import("@tensorflow/tfjs-backend-wasm");
  wasm.setWasmPaths("/tfjs-wasm/");
  for (const name of ["webgl", "wasm", "cpu"]) {
    try {
      if (await tf.setBackend(name)) {
        await tf.ready();
        return name;
      }
    } catch {
      // try the next backend
    }
  }
  throw new Error("No TensorFlow.js backend is available in this browser.");
}

/** Load once; later calls reuse the same model. ``onProgress`` receives 0..1. */
export function loadModel(onProgress?: (fraction: number) => void): Promise<Loaded> {
  if (!loading) {
    loading = (async () => {
      const tf = await import("@tensorflow/tfjs");
      const backend = await pickBackend(tf);
      const [model, info] = await Promise.all([
        tf.loadGraphModel(MODEL_URL, { onProgress }),
        fetch(MODEL_INFO_URL).then((r) => {
          if (!r.ok) throw new Error(`Could not load ${MODEL_INFO_URL} (${r.status}).`);
          return r.json() as Promise<ModelInfo>;
        }),
      ]);
      // warm-up so the first real prediction is not slowed by shader compilation
      tf.tidy(() => model.predict(tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3])));
      return { tf, model, backend, info };
    })().catch((e) => {
      loading = null; // allow a retry
      throw e;
    });
  }
  return loading;
}

export type InferenceResult = { probs: Float32Array; backend: string; ms: number };

/**
 * Classify an image. Resizing uses bilinear interpolation with half-pixel centres, which
 * matches tf.image.resize(method="bilinear") used in training. The probabilities are averaged
 * over four views of the image (test-time augmentation, lib/tta.ts), as in the evaluation.
 */
export async function classify(source: HTMLImageElement | HTMLCanvasElement): Promise<InferenceResult> {
  const { tf, model, backend, info } = await loadModel();
  const t0 = performance.now();
  const out = tf.tidy(() => {
    const pixels = tf.browser.fromPixels(source).toFloat();
    const resized = tf.image.resizeBilinear(pixels, [INPUT_SIZE, INPUT_SIZE], false, true);
    const x = resized.expandDims(0) as TF.Tensor4D;
    return info.tta_views === 4 ? predictTta(tf, model, x) : (model.predict(x) as TF.Tensor);
  });
  const probs = (await out.data()) as Float32Array;
  out.dispose();
  return { probs, backend, ms: performance.now() - t0 };
}

export type Explanation = {
  /** Heat map in [0, 1], INPUT_SIZE x INPUT_SIZE, row-major. */
  map: Float32Array;
  /** The class the map explains, and the model's probability for it on this image. */
  classIndex: number;
  probability: number;
  ms: number;
};

/**
 * Grad-CAM for ``classIndex``, computed on the device (see lib/gradcam.ts): one forward pass
 * returns the last feature maps and the probabilities, and the dense layer's weights are read
 * from the loaded model. No gradients and no server are needed. The map explains the original
 * view of the image (the first of the four test-time views), so ``probability`` is that view's.
 */
export async function explain(source: HTMLImageElement | HTMLCanvasElement, classIndex: number): Promise<Explanation> {
  const { tf, model, info } = await loadModel();
  const t0 = performance.now();
  const [feats, probs] = tf.tidy(() => {
    const pixels = tf.browser.fromPixels(source).toFloat();
    const resized = tf.image.resizeBilinear(pixels, [INPUT_SIZE, INPUT_SIZE], false, true);
    return model.execute(resized.expandDims(0), [info.feature_node, info.output_node]) as TF.Tensor[];
  });
  const kernel = model.weights[info.kernel_weight]?.[0];
  if (!kernel) {
    tf.dispose([feats, probs]);
    throw new Error("The model file does not contain the classifier weights.");
  }
  const [, h, w, k] = feats.shape;
  const [f, p, wk] = await Promise.all([feats.data(), probs.data(), kernel.data()]);
  tf.dispose([feats, probs]);
  const { map } = gradcam(f, { h, w, k }, p, wk, classIndex, { h: INPUT_SIZE, w: INPUT_SIZE });
  return { map, classIndex, probability: p[classIndex], ms: performance.now() - t0 };
}
