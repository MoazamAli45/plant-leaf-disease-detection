/**
 * The in-browser Grad-CAM (closed form, float16 TF.js model) against reference maps computed
 * with TensorFlow's automatic gradient on the float32 Keras model.
 *
 * Needs the exported test data, which is not committed:
 *   (cd ../ml && .venv/bin/python -m leafdx.train.export_web verify)
 * Skipped when it is missing (e.g. in CI). Writes test-data/gradcam-result.json.
 */
import fs from "node:fs";
import path from "node:path";

import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import { describe, expect, it } from "vitest";

import { FEATURE_NODE, KERNEL_WEIGHT, OUTPUT_NODE, gradcam } from "@/lib/gradcam";

const root = path.resolve(__dirname, "../../..");
const data = path.join(root, "test-data");
const modelDir = path.join(root, "public", "model");
const available = ["images.f32", "expected.json", "gradcam.json"].every((f) => fs.existsSync(path.join(data, f)));

function loadModel() {
  const json = JSON.parse(fs.readFileSync(path.join(modelDir, "model.json"), "utf8"));
  const all = Buffer.concat(json.weightsManifest.flatMap((g: { paths: string[] }) => g.paths.map((p) => fs.readFileSync(path.join(modelDir, p)))));
  return tf.loadGraphModel(
    tf.io.fromMemory({
      modelTopology: json.modelTopology,
      weightSpecs: json.weightsManifest.flatMap((g: { weights: unknown[] }) => g.weights),
      weightData: all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength),
      format: json.format,
      signature: json.signature,
    }),
  );
}

function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i] / n;
    mb += b[i] / n;
  }
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 1;
}

describe.skipIf(!available)("in-browser Grad-CAM vs Keras GradientTape", () => {
  it("matches the reference maps on the exported test images", async () => {
    setWasmPaths(path.join(root, "node_modules", "@tensorflow", "tfjs-backend-wasm", "dist") + "/");
    await tf.setBackend("wasm");
    await tf.ready();
    const model = await loadModel();
    const expected = JSON.parse(fs.readFileSync(path.join(data, "expected.json"), "utf8"));
    const ref = JSON.parse(fs.readFileSync(path.join(data, "gradcam.json"), "utf8")) as { class_index: number[]; cams: number[][][] };
    const [n, h, w, c] = expected.shape as number[];
    const raw = fs.readFileSync(path.join(data, "images.f32"));
    const images = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    const kernel = model.weights[KERNEL_WEIGHT][0].dataSync();

    const corr: number[] = [];
    const maxDiff: number[] = [];
    const samePeak: boolean[] = [];
    for (let i = 0; i < n; i++) {
      const x = tf.tensor4d(images.subarray(i * h * w * c, (i + 1) * h * w * c), [1, h, w, c]);
      const [feats, probs] = model.execute(x, [FEATURE_NODE, OUTPUT_NODE]) as tf.Tensor[];
      const [, fh, fw, k] = feats.shape;
      const [fd, pd] = await Promise.all([feats.data(), probs.data()]);
      const { small } = gradcam(fd, { h: fh, w: fw, k }, pd, kernel, ref.class_index[i], { h: fh, w: fw });
      tf.dispose([x, feats, probs]);
      const r = ref.cams[i].flat();
      corr.push(pearson(small, r));
      maxDiff.push(Math.max(...r.map((v, j) => Math.abs(v - small[j]))));
      const argmax = (a: ArrayLike<number>) => Array.from(a).indexOf(Math.max(...Array.from(a)));
      samePeak.push(argmax(small) === argmax(r));
    }
    const sorted = [...corr].sort((a, b) => a - b);
    const result = {
      images: n,
      reference: "Keras float32 model, tf.GradientTape (ml/src/leafdx/train/export_web.py)",
      browser: "TF.js float16 model, closed form (web/src/lib/gradcam.ts), wasm backend",
      resolution: "7 x 7, each map scaled to a maximum of 1",
      pearson_median: sorted[Math.floor(n / 2)],
      pearson_min: sorted[0],
      max_abs_diff_median: [...maxDiff].sort((a, b) => a - b)[Math.floor(n / 2)],
      max_abs_diff_max: Math.max(...maxDiff),
      same_hottest_cell: samePeak.filter(Boolean).length / n,
    };
    fs.writeFileSync(path.join(data, "gradcam-result.json"), JSON.stringify(result, null, 1) + "\n");
    console.log(result);

    expect(result.pearson_median).toBeGreaterThan(0.99);
    expect(result.pearson_min).toBeGreaterThan(0.9);
  }, 300_000);
});
