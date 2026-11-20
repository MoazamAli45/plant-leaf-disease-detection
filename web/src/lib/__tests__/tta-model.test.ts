/**
 * Test-time augmentation in the browser (lib/tta.ts, float16 TF.js model) against the Python
 * evaluation (ml/src/leafdx/field/methods.py, float32 Keras model) on the exported test images.
 *
 * Needs the exported test data, which is not committed:
 *   (cd ../ml && .venv/bin/python -m leafdx.train.export_web verify --model <run>/model.keras)
 * Skipped when it is missing (e.g. in CI). Writes test-data/tta-result.json.
 */
import fs from "node:fs";
import path from "node:path";

import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import { beforeAll, describe, expect, it } from "vitest";

import { predictTta, ttaViews } from "@/lib/tta";

const root = path.resolve(__dirname, "../../..");
const data = path.join(root, "test-data");
const modelDir = path.join(root, "public", "model");
const expectedFile = path.join(data, "expected.json");
const available =
  fs.existsSync(path.join(data, "images.f32")) &&
  fs.existsSync(expectedFile) &&
  "probs_tta" in JSON.parse(fs.readFileSync(expectedFile, "utf8"));

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

beforeAll(async () => {
  setWasmPaths(path.join(root, "node_modules", "@tensorflow", "tfjs-backend-wasm", "dist") + "/");
  await tf.setBackend("wasm");
  await tf.ready();
});

const argmax = (a: ArrayLike<number>) => Array.from(a).indexOf(Math.max(...Array.from(a)));

describe("test-time augmentation views", () => {
  it("returns the image, its flip, the 87.5% centre crop and the crop's flip", () => {
    const x = tf.tensor4d(Array.from({ length: 8 * 8 * 3 }, (_, i) => i), [1, 8, 8, 3]);
    const v = ttaViews(tf, x);
    expect(v.shape).toEqual([4, 8, 8, 3]);
    expect(Array.from(v.slice([0], [1]).dataSync())).toEqual(Array.from(x.dataSync()));
    expect(Array.from(v.slice([1], [1]).dataSync())).toEqual(Array.from(tf.reverse(x, 2).dataSync()));
    expect(Array.from(v.slice([3], [1]).dataSync())).toEqual(Array.from(tf.reverse(v.slice([2], [1]), 2).dataSync()));
    tf.dispose([x, v]);
  });
});

describe.skipIf(!available)("in-browser test-time augmentation vs Python", () => {
  it("gives the same averaged probabilities on the exported test images", async () => {
    const model = await loadModel();
    const expected = JSON.parse(fs.readFileSync(expectedFile, "utf8"));
    const [n, h, w, c] = expected.shape as number[];
    const raw = fs.readFileSync(path.join(data, "images.f32"));
    const images = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);

    let agree = 0;
    let maxDiff = 0;
    for (let i = 0; i < n; i++) {
      const x = tf.tensor4d(images.subarray(i * h * w * c, (i + 1) * h * w * c), [1, h, w, c]);
      const y = predictTta(tf, model, x);
      const p = await y.data();
      tf.dispose([x, y]);
      const ref = expected.probs_tta[i] as number[];
      for (let k = 0; k < p.length; k++) maxDiff = Math.max(maxDiff, Math.abs(p[k] - ref[k]));
      if (argmax(p) === argmax(ref)) agree++;
    }
    const result = {
      images: n,
      reference: "Keras float32 model, 4 views (ml/src/leafdx/field/methods.py)",
      browser: "TF.js float16 model, 4 views (web/src/lib/tta.ts), wasm backend",
      same_top1: agree / n,
      max_abs_prob_diff: maxDiff,
    };
    fs.writeFileSync(path.join(data, "tta-result.json"), JSON.stringify(result, null, 1) + "\n");
    console.log(result);
    expect(result.same_top1).toBeGreaterThanOrEqual(0.98);
    expect(result.max_abs_prob_diff).toBeLessThan(0.05);
  }, 300_000);
});
