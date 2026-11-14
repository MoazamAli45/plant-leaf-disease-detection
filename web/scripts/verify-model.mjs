// Checks that the TF.js model in public/model gives the same answers as the Keras model, and
// measures single-image latency with the TF.js CPU and WebAssembly backends in Node.
//
//   (cd ../ml && .venv/bin/python -m leafdx.train.export_web verify)   # writes test-data/
//   node scripts/verify-model.mjs [modelDir] [outFile] [backends]
//   e.g. node scripts/verify-model.mjs ../ml/models/_export/p3_mobilenet_v1/tfjs_uint8 out.json wasm
//
// Writes test-data/verify-result.json. WebGL (the usual browser backend) cannot run in Node;
// the app reports its own latency in the browser.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const modelDir = path.resolve(process.argv[2] ?? path.join(root, "public", "model"));
const outFile = path.resolve(process.argv[3] ?? path.join(root, "test-data", "verify-result.json"));
const backends = (process.argv[4] ?? "wasm,cpu").split(",");
const dataDir = path.join(root, "test-data");

function loadArtifacts() {
  const json = JSON.parse(fs.readFileSync(path.join(modelDir, "model.json"), "utf8"));
  const buffers = json.weightsManifest.flatMap((g) =>
    g.paths.map((p) => fs.readFileSync(path.join(modelDir, p))),
  );
  const all = Buffer.concat(buffers);
  return tf.io.fromMemory({
    modelTopology: json.modelTopology,
    weightSpecs: json.weightsManifest.flatMap((g) => g.weights),
    weightData: all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength),
    format: json.format,
    generatedBy: json.generatedBy,
    convertedBy: json.convertedBy,
    signature: json.signature,
    userDefinedMetadata: json.userDefinedMetadata,
  });
}

function argmax(a) {
  let k = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i;
  return k;
}

async function run(backend) {
  await tf.setBackend(backend);
  await tf.ready();
  const model = await tf.loadGraphModel(loadArtifacts());
  const expected = JSON.parse(fs.readFileSync(path.join(dataDir, "expected.json"), "utf8"));
  const [n, h, w, c] = expected.shape;
  const raw = fs.readFileSync(path.join(dataDir, "images.f32"));
  const all = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);

  let agree = 0;
  let maxDiff = 0;
  const correct = { test: [0, 0], plantdoc: [0, 0] };
  for (let i = 0; i < n; i++) {
    const x = tf.tensor4d(all.subarray(i * h * w * c, (i + 1) * h * w * c), [1, h, w, c]);
    const y = model.predict(x);
    const p = await y.data();
    tf.dispose([x, y]);
    const ref = expected.probs[i];
    for (let k = 0; k < p.length; k++) maxDiff = Math.max(maxDiff, Math.abs(p[k] - ref[k]));
    if (argmax(p) === argmax(ref)) agree++;
    const s = correct[expected.sets[i]];
    s[1]++;
    if (argmax(p) === expected.labels[i]) s[0]++;
  }

  // latency: batch 1, after warm-up
  const x = tf.tensor4d(all.subarray(0, h * w * c), [1, h, w, c]);
  for (let i = 0; i < 5; i++) tf.dispose(model.predict(x));
  const times = [];
  for (let i = 0; i < 30; i++) {
    const t0 = performance.now();
    const y = model.predict(x);
    await y.data();
    times.push(performance.now() - t0);
    tf.dispose(y);
  }
  x.dispose();
  times.sort((a, b) => a - b);
  return {
    backend,
    images: n,
    agreement_with_keras: agree / n,
    max_abs_prob_diff: maxDiff,
    accuracy_test: correct.test[0] / correct.test[1],
    accuracy_plantdoc: correct.plantdoc[0] / correct.plantdoc[1],
    n_test: correct.test[1],
    n_plantdoc: correct.plantdoc[1],
    latency_ms_median: times[Math.floor(times.length / 2)],
    latency_ms_p90: times[Math.floor(times.length * 0.9)],
  };
}

setWasmPaths(path.join(root, "node_modules", "@tensorflow", "tfjs-backend-wasm", "dist") + "/");
const results = [];
for (const backend of backends) {
  try {
    const r = await run(backend);
    console.log(JSON.stringify(r));
    results.push(r);
  } catch (e) {
    console.log(`${backend}: failed: ${e.message}`);
  }
}
const expectedAcc = (() => {
  const e = JSON.parse(fs.readFileSync(path.join(dataDir, "expected.json"), "utf8"));
  const ok = e.probs.map((p, i) => argmax(p) === e.labels[i]);
  const acc = (set) => {
    const idx = e.sets.map((s, i) => (s === set ? i : -1)).filter((i) => i >= 0);
    return idx.filter((i) => ok[i]).length / idx.length;
  };
  return { test: acc("test"), plantdoc: acc("plantdoc") };
})();
fs.writeFileSync(
  outFile,
  JSON.stringify(
    { model_dir: path.relative(path.dirname(root), modelDir), node: process.version,
      tfjs: tf.version.tfjs, keras_accuracy: expectedAcc, results },
    null,
    1,
  ),
);
console.log("keras accuracy on the same images:", JSON.stringify(expectedAcc));
const bad = results.some((r) => r.agreement_with_keras < 0.98);
process.exit(bad || results.length === 0 ? 1 : 0);
