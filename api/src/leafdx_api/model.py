"""Model loading, prediction and Grad-CAM.

The model is the float32 inference export of the deployed MobileNetV1
(``ml/models/_export/p3_mobilenet_v1/inference.keras``): raw [0, 255] RGB in, 38 probabilities
out. Grad-CAM uses the last convolutional block (the backbone for transfer models).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


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

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model(x, training=False))[0]

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
