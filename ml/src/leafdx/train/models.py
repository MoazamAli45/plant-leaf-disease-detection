"""The two models of Usman & Ali, as described in the paper (Figures 2 and 3).

Paper CNN (Figure 2): three blocks of Conv2D(32, 3x3, ReLU) -> MaxPool(2x2) -> Dropout(0.2),
then Conv2D(128, 3x3, ReLU) -> MaxPool(2x2), Flatten, Dense(128, ReLU), Dense(64, ReLU),
Dense(38, sigmoid). Padding is not stated; Keras' default ("valid") is assumed.

Paper MobileNet (Figure 3): ImageNet-pretrained MobileNet (V1) base, GlobalAveragePooling,
Dropout(0.2), Dense(38, softmax).
"""

from __future__ import annotations

import keras
from keras import layers

from leafdx.train.config import ModelCfg


def build_paper_cnn(cfg: ModelCfg, image_size: int) -> keras.Model:
    inputs = keras.Input((image_size, image_size, 3), name="image")
    x = inputs
    for i, filters in enumerate((32, 32, 32), start=1):
        x = layers.Conv2D(filters, 3, activation="relu", name=f"conv{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
        x = layers.Dropout(cfg.dropout, name=f"drop{i}")(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv4")(x)
    x = layers.MaxPooling2D(2, name="pool4")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dense(64, activation="relu", name="dense2")(x)
    outputs = layers.Dense(
        cfg.num_classes, activation=cfg.output_activation, dtype="float32", name="dense3"
    )(x)
    return keras.Model(inputs, outputs, name="paper_cnn")


def build_paper_mobilenet(cfg: ModelCfg, image_size: int) -> keras.Model:
    base = keras.applications.MobileNet(
        input_shape=(image_size, image_size, 3),
        alpha=cfg.alpha,
        include_top=False,
        weights=cfg.weights,
    )
    base.trainable = not cfg.freeze_backbone
    inputs = keras.Input((image_size, image_size, 3), name="image")
    # training=False keeps BatchNorm in inference mode when the base is frozen (Keras guidance
    # for transfer learning); when fine-tuning the base, BN statistics are updated.
    x = base(inputs, training=False if cfg.freeze_backbone else None)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(cfg.dropout, name="dropout")(x)
    outputs = layers.Dense(
        cfg.num_classes, activation=cfg.output_activation, dtype="float32", name="classifier"
    )(x)
    return keras.Model(inputs, outputs, name="paper_mobilenet")


BUILDERS = {"paper_cnn": build_paper_cnn, "paper_mobilenet": build_paper_mobilenet}


def build_model(cfg: ModelCfg, image_size: int) -> keras.Model:
    try:
        builder = BUILDERS[cfg.name]
    except KeyError as e:
        raise ValueError(f"unknown model {cfg.name!r}; choose from {sorted(BUILDERS)}") from e
    return builder(cfg, image_size)
