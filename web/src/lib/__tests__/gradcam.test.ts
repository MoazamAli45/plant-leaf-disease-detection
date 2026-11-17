import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it } from "vitest";

import { channelWeights, colourise, combine, gradcam, normalise, resizeBilinear } from "@/lib/gradcam";

beforeAll(async () => {
  await tf.setBackend("cpu");
  await tf.ready();
});

/** Deterministic pseudo-random numbers (mulberry32). */
function rng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("closed-form Grad-CAM weights", () => {
  it("equal the spatial mean of the automatic gradient for GAP -> dense -> softmax", () => {
    const [h, w, k, c] = [3, 4, 6, 5];
    const r = rng(1);
    const feats = Float32Array.from({ length: h * w * k }, () => r() * 3);
    const kernel = Float32Array.from({ length: k * c }, () => r() * 2 - 1);
    const bias = Float32Array.from({ length: c }, () => r() - 0.5);

    const head = (a: tf.Tensor4D) => tf.softmax(tf.add(tf.matMul(tf.mean(a, [1, 2]), tf.tensor2d(kernel, [k, c])), tf.tensor1d(bias)));
    const A = tf.tensor4d(feats, [1, h, w, k]);
    const probs = head(A).dataSync();

    for (let cls = 0; cls < c; cls++) {
      const grad = tf.grad((a: tf.Tensor) => head(a as tf.Tensor4D).gather([cls], 1).sum())(A);
      const expected = tf.mean(grad, [0, 1, 2]).dataSync();
      const got = channelWeights(probs, kernel, k, c, cls, h * w);
      for (let f = 0; f < k; f++) expect(got[f]).toBeCloseTo(expected[f], 6);
    }
  });
});

describe("gradcam pipeline", () => {
  it("combines with ReLU and normalises to a maximum of 1", () => {
    // 2 positions, 2 channels: position 0 -> 1*2 + 0*1 = 2, position 1 -> -1*1 + 0 = negative
    const cam = combine([2, 1, 0, -1], Float32Array.from([1, 0]), 2);
    expect(Array.from(cam)).toEqual([2, 0]);
    expect(Array.from(normalise(cam))).toEqual([1, 0]);
    expect(Array.from(normalise(new Float32Array(3)))).toEqual([0, 0, 0]);
  });

  it("resizes like tf.image.resizeBilinear with half-pixel centres", () => {
    const r = rng(2);
    const src = Float32Array.from({ length: 7 * 7 }, () => r());
    const ours = resizeBilinear(src, 7, 7, 224, 224);
    const ref = tf.image.resizeBilinear(tf.tensor3d(src, [7, 7, 1]), [224, 224], false, true).dataSync();
    let maxDiff = 0;
    for (let i = 0; i < ours.length; i++) maxDiff = Math.max(maxDiff, Math.abs(ours[i] - ref[i]));
    expect(maxDiff).toBeLessThan(1e-5);
  });

  it("returns maps in [0, 1] at the requested size", () => {
    const r = rng(3);
    const [h, w, k, c] = [7, 7, 8, 4];
    const feats = Float32Array.from({ length: h * w * k }, () => r());
    const kernel = Float32Array.from({ length: k * c }, () => r() * 2 - 1);
    const probs = [0.1, 0.6, 0.2, 0.1];
    const { small, map } = gradcam(feats, { h, w, k }, probs, kernel, 1, { h: 32, w: 32 });
    expect(small.length).toBe(49);
    expect(map.length).toBe(32 * 32);
    expect(Math.max(...map)).toBeCloseTo(1, 6);
    expect(Math.min(...map)).toBeGreaterThanOrEqual(0);
  });

  it("colours 0 as near-black and 1 as pale yellow, fully opaque", () => {
    const px = colourise(Float32Array.from([0, 1]));
    expect(Array.from(px.slice(0, 4))).toEqual([0, 0, 4, 255]);
    expect(Array.from(px.slice(4, 8))).toEqual([252, 255, 164, 255]);
  });
});
