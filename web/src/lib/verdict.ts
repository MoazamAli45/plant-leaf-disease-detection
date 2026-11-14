/**
 * Turning model probabilities into what the app tells the user.
 *
 * Thresholds come from the Phase 3 selective-prediction analysis
 * (docs/figures/phase3/selective_prediction.md): for MobileNetV1 on lab test images, answering
 * only when the top probability is at least 0.7 keeps 98.8% of images at 99.2% accuracy.
 * On field photos (PlantDoc) the same model is right only about 20% of the time even when
 * confident, so the app always shows a field-photo caveat. It never claims certainty.
 */

export const CONFIDENT = 0.7;
/** Below this the app says "not sure" instead of "low confidence" (a wording choice). */
export const NOT_SURE = 0.4;
/** Below this share of plant-coloured pixels the photo probably does not show a leaf. */
export const MIN_PLANT_PIXELS = 0.12;

export type Verdict = "confident" | "uncertain" | "not-sure" | "not-a-leaf";

export type Prediction = { index: number; probability: number };

export function topK(probs: ArrayLike<number>, k = 3): Prediction[] {
  return Array.from(probs, (p, index) => ({ index, probability: p }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, k);
}

export function verdict(top: Prediction[], plantPixelShare: number): Verdict {
  if (plantPixelShare < MIN_PLANT_PIXELS) return "not-a-leaf";
  const p = top[0]?.probability ?? 0;
  if (p >= CONFIDENT) return "confident";
  if (p >= NOT_SURE) return "uncertain";
  return "not-sure";
}

/**
 * Share of pixels that look like plant tissue: saturated colours with a hue from yellow-brown
 * to green (about 20 to 170 degrees). A heuristic, not a leaf detector: it catches photos
 * of grey backgrounds, people or screens, but a green object will pass.
 */
export function plantPixelShare(rgba: Uint8ClampedArray): number {
  let plant = 0;
  let n = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    const r = rgba[i] / 255;
    const g = rgba[i + 1] / 255;
    const b = rgba[i + 2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    n++;
    if (max < 0.12 || d / (max || 1) < 0.18) continue; // too dark or too grey
    let h: number;
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    const hue = (h * 60 + 360) % 360;
    if (hue >= 20 && hue <= 170) plant++;
  }
  return n ? plant / n : 0;
}
