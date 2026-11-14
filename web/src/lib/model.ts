"use client";

/**
 * In-browser inference with TensorFlow.js. The model (MobileNetV1, float16 weights, 6.3 MB)
 * takes raw RGB pixels in [0, 255] at 224 x 224; preprocessing is inside the model. Images
 * never leave the device.
 *
 * Verified against the Keras model in Node (docs/figures/phase3/tfjs_verify_mnv1_tfjs_fp16.json):
 * same top-1 class on 100/100 images, max probability difference 0.011.
 */

import type * as TF from "@tensorflow/tfjs";

export const MODEL_URL = "/model/model.json";
export const INPUT_SIZE = 224;

type Loaded = { tf: typeof TF; model: TF.GraphModel; backend: string };

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
      const model = await tf.loadGraphModel(MODEL_URL, { onProgress });
      // warm-up so the first real prediction is not slowed by shader compilation
      tf.tidy(() => model.predict(tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3])));
      return { tf, model, backend };
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
 * matches tf.image.resize(method="bilinear") used in training.
 */
export async function classify(source: HTMLImageElement | HTMLCanvasElement): Promise<InferenceResult> {
  const { tf, model, backend } = await loadModel();
  const t0 = performance.now();
  const out = tf.tidy(() => {
    const pixels = tf.browser.fromPixels(source).toFloat();
    const resized = tf.image.resizeBilinear(pixels, [INPUT_SIZE, INPUT_SIZE], false, true);
    return model.predict(resized.expandDims(0)) as TF.Tensor;
  });
  const probs = (await out.data()) as Float32Array;
  out.dispose();
  return { probs, backend, ms: performance.now() - t0 };
}
