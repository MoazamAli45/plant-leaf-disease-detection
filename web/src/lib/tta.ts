import type * as TF from "@tensorflow/tfjs";

/**
 * Test-time augmentation, the same four views as the Python evaluation
 * (ml/src/leafdx/field/methods.py, ``_views``): the image, its horizontal flip, an 87.5% centre
 * crop resized back to full size, and the crop's flip. The model's probabilities are averaged
 * over the views (docs/field.md, Phase 6).
 */
export const TTA_VIEWS = 4;
export const CROP_FRACTION = 0.875;

/** ``x``: [1, size, size, 3] -> [4, size, size, 3]. */
export function ttaViews(tf: typeof TF, x: TF.Tensor4D): TF.Tensor4D {
  return tf.tidy(() => {
    const [, h, w] = x.shape;
    // tf.image.central_crop: offset int((h - h * f) / 2), size h - 2 * offset
    const top = Math.floor((h - h * CROP_FRACTION) / 2);
    const left = Math.floor((w - w * CROP_FRACTION) / 2);
    const crop = tf.image.resizeBilinear(
      tf.slice(x, [0, top, left, 0], [-1, h - 2 * top, w - 2 * left, -1]),
      [h, w],
      false,
      true, // half-pixel centres, as tf.image.resize(method="bilinear")
    );
    return tf.concat([x, tf.image.flipLeftRight(x), crop, tf.image.flipLeftRight(crop)], 0);
  });
}

/** Average the model's probabilities over the four views. */
export function predictTta(tf: typeof TF, model: TF.GraphModel, x: TF.Tensor4D): TF.Tensor2D {
  return tf.tidy(() => (model.predict(ttaViews(tf, x)) as TF.Tensor2D).mean(0, true));
}
