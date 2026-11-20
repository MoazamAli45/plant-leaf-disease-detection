/**
 * Grad-CAM computed in the browser, without automatic differentiation.
 *
 * The deployed model ends in global average pooling -> dropout (inactive at inference) ->
 * dense -> softmax. For that head the gradient of the class probability p_c with respect to
 * the last feature maps A (H x W x K) has a closed form:
 *
 *   dp_c / dA_k(u, v) = p_c * (W[k, c] - sum_j p_j * W[k, j]) / (H * W)
 *
 * It is the same at every position, so it is also the Grad-CAM weight alpha_k (the spatial mean
 * of the gradient). The map is ReLU(sum_k alpha_k * A_k), resized bilinearly and scaled to a
 * maximum of 1: exactly what the Python implementation computes with tf.GradientTape
 * (ml/src/leafdx/explain/gradcam.py). scripts/verify-model.mjs checks the two agree.
 *
 * Everything here is plain arithmetic on typed arrays, so it can be unit-tested without a
 * browser.
 */

/**
 * What the app needs to know about the deployed model, from web/public/model/leafdx.json
 * (written by ml/src/leafdx/field/deploy.py, which finds the node names in the converted graph).
 */
export type ModelInfo = {
  /** Backbone name shown to users, e.g. "MobileNetV1" or "EfficientNetB0". */
  label: string;
  /** Last feature maps, the input of the global average pooling. */
  feature_node: string;
  /** The classifier's K x C dense kernel. */
  kernel_weight: string;
  /** The softmax probabilities. */
  output_node: string;
  /** 4 = test-time augmentation (lib/tta.ts), 1 = a single view. */
  tta_views: number;
};

/** Grad-CAM weight per feature map for class ``c``. ``kernel`` is the dense layer's
 * K x C weight matrix in row-major order. */
export function channelWeights(probs: ArrayLike<number>, kernel: ArrayLike<number>, k: number, c: number, cls: number, hw: number): Float32Array {
  const out = new Float32Array(k);
  const pc = probs[cls];
  for (let f = 0; f < k; f++) {
    let expected = 0;
    const row = f * c;
    for (let j = 0; j < c; j++) expected += probs[j] * kernel[row + j];
    out[f] = (pc * (kernel[row + cls] - expected)) / hw;
  }
  return out;
}

/** ReLU(sum_k alpha_k * A_k) for features in H x W x K (row-major, channels last). */
export function combine(features: ArrayLike<number>, alpha: Float32Array, hw: number): Float32Array {
  const k = alpha.length;
  const cam = new Float32Array(hw);
  for (let p = 0; p < hw; p++) {
    let s = 0;
    const base = p * k;
    for (let f = 0; f < k; f++) s += alpha[f] * features[base + f];
    cam[p] = s > 0 ? s : 0;
  }
  return cam;
}

/** Scale to a maximum of 1 (all zeros stay zeros). */
export function normalise(cam: Float32Array): Float32Array {
  let peak = 0;
  for (const v of cam) if (v > peak) peak = v;
  return peak > 0 ? cam.map((v) => v / peak) : cam;
}

/** Bilinear resize with half-pixel centres and edge clamping (tf.image.resize, PIL BILINEAR
 * when upsampling). */
export function resizeBilinear(src: Float32Array, h: number, w: number, outH: number, outW: number): Float32Array {
  const out = new Float32Array(outH * outW);
  const sy = h / outH;
  const sx = w / outW;
  for (let y = 0; y < outH; y++) {
    const fy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), h - 1);
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, h - 1);
    const ty = fy - y0;
    for (let x = 0; x < outW; x++) {
      const fx = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), w - 1);
      const x0 = Math.floor(fx);
      const x1 = Math.min(x0 + 1, w - 1);
      const tx = fx - x0;
      const top = src[y0 * w + x0] * (1 - tx) + src[y0 * w + x1] * tx;
      const bottom = src[y1 * w + x0] * (1 - tx) + src[y1 * w + x1] * tx;
      out[y * outW + x] = top * (1 - ty) + bottom * ty;
    }
  }
  return out;
}

/** Full Grad-CAM: features (H x W x K), probabilities and the dense kernel (K x C) in; a map in
 * [0, 1] of size outH x outW out. */
export function gradcam(
  features: ArrayLike<number>,
  shape: { h: number; w: number; k: number },
  probs: ArrayLike<number>,
  kernel: ArrayLike<number>,
  cls: number,
  out: { h: number; w: number },
): { small: Float32Array; map: Float32Array } {
  const hw = shape.h * shape.w;
  const alpha = channelWeights(probs, kernel, shape.k, probs.length, cls, hw);
  const small = normalise(combine(features, alpha, hw));
  return { small, map: normalise(resizeBilinear(small, shape.h, shape.w, out.h, out.w)) };
}

// Same colour scale as the API's overlay (a compact "inferno"): black -> purple -> red ->
// orange -> pale yellow.
const STOPS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0, 4],
  [87, 16, 110],
  [188, 55, 84],
  [249, 142, 9],
  [252, 255, 164],
];

/** RGBA pixels (opaque) for a map in [0, 1]. */
export function colourise(map: Float32Array): Uint8ClampedArray<ArrayBuffer> {
  const px = new Uint8ClampedArray(new ArrayBuffer(map.length * 4));
  const n = STOPS.length - 1;
  for (let i = 0; i < map.length; i++) {
    const pos = Math.min(Math.max(map[i], 0), 1) * n;
    const lo = Math.min(Math.floor(pos), n - 1);
    const t = pos - lo;
    const a = STOPS[lo];
    const b = STOPS[lo + 1];
    px[i * 4] = a[0] + (b[0] - a[0]) * t;
    px[i * 4 + 1] = a[1] + (b[1] - a[1]) * t;
    px[i * 4 + 2] = a[2] + (b[2] - a[2]) * t;
    px[i * 4 + 3] = 255;
  }
  return px;
}
