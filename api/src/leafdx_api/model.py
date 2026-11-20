"""Model loading, prediction and Grad-CAM.

The model is the float32 Keras file of the deployed MobileNetV1, fine-tuned on field photos
(``ml/models/_export/p6_deployed/inference.keras``, docs/field.md): raw [0, 255] RGB in, 38
probabilities out. Predictions average four views (test-time augmentation), exactly as the
evaluation and the web app do. Grad-CAM uses the last convolutional block of the original
view (the backbone for transfer models).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

CROP_FRACTION = 0.875


def tta_views(x: tf.Tensor) -> tf.Tensor:
    """[1, H, W, 3] -> [4, H, W, 3]: the image, its horizontal flip, an 87.5% centre crop
    resized back, and its flip (as ``ml/src/leafdx/field/methods.py``, ``_views``)."""
    size = x.shape[1:3]
    crop = tf.image.resize(tf.image.central_crop(x, CROP_FRACTION), size, method="bilinear")
    return tf.concat([x, tf.image.flip_left_right(x), crop, tf.image.flip_left_right(crop)], 0)


def target_layer(model: keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model | keras.layers.Conv2D):
            return layer.name
    raise ValueError("model has no convolutional layer for Grad-CAM")


def gradcam_model(model: keras.Model) -> keras.Model:
    """(features, probabilities). The graph is rebuilt by calling the layers in order, which
    works for linear models whose Grad-CAM target is a nested backbone."""
    target = target_layer(model)
    inp = keras.Input(model.inputs[0].shape[1:])
    h, feats = inp, None
    for layer in model.layers[1:]:
        h = layer(h)
        if layer.name == target:
            feats = h
    return keras.Model(inp, [feats, h])


@dataclass
class Classifier:
    model: keras.Model
    grad: keras.Model
    classes: list[dict]

    @classmethod
    def load(cls, model_path: Path, classes_path: Path) -> Classifier:
        model = keras.models.load_model(model_path, compile=False)
        classes = json.loads(classes_path.read_text())
        n_out = int(model.outputs[0].shape[-1])
        if n_out != len(classes):
            raise ValueError(f"model has {n_out} outputs but {len(classes)} classes are listed")
        return cls(model=model, grad=gradcam_model(model), classes=classes)

    def predict(self, x: np.ndarray, tta: bool = True) -> np.ndarray:
        x = tf.convert_to_tensor(x, tf.float32)
        views = tta_views(x) if tta else x
        return np.asarray(self.model(views, training=False)).mean(axis=0)

    def gradcam(self, x: np.ndarray, class_index: int) -> np.ndarray:
        """Heat map in [0, 1] at the input resolution."""
        inp = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            feats, probs = self.grad(inp, training=False)
            score = probs[:, class_index]
        grads = tape.gradient(score, feats)
        feats = tf.cast(feats, tf.float32)
        grads = tf.cast(grads, tf.float32)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * feats, axis=-1))[0]
        cam = tf.image.resize(cam[..., None], x.shape[1:3])[..., 0].numpy()
        peak = cam.max()
        return cam / peak if peak > 0 else cam
