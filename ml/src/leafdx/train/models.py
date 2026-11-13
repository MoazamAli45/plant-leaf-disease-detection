"""Models.

Paper CNN (Figure 2 of Usman & Ali): three blocks of Conv2D(32, 3x3, ReLU) -> MaxPool(2x2) ->
Dropout(0.2), then Conv2D(128, 3x3, ReLU) -> MaxPool(2x2), Flatten, Dense(128, ReLU),
Dense(64, ReLU), Dense(38, sigmoid). Padding is not stated; Keras' default ("valid") is assumed.

Paper MobileNet (Figure 3): ImageNet-pretrained MobileNet (V1) base, GlobalAveragePooling,
Dropout(0.2), Dense(38, softmax).

Transfer models (Phase 3): the same head on MobileNetV1/V2/V3-Small or EfficientNetB0.

All builders take an optional ``stem``: layers applied right after the input (augmentation and
preprocessing when ``data.augment_in_model`` is set). Augmentation layers are only active in
training, so the saved model maps raw [0, 255] pixels to class probabilities.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import keras
from keras import layers

from leafdx.train.config import ModelCfg

Stem = Sequence[keras.layers.Layer] | None


def _input_and_stem(image_size: int, stem: Stem) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    inputs = keras.Input((image_size, image_size, 3), name="image")
    x = inputs
    for layer in stem or ():
        x = layer(x)
    return inputs, x


def build_paper_cnn(cfg: ModelCfg, image_size: int, stem: Stem = None) -> keras.Model:
    inputs, x = _input_and_stem(image_size, stem)
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


def _backbone(cfg: ModelCfg, image_size: int) -> keras.Model:
    shape = (image_size, image_size, 3)
    common = {"input_shape": shape, "include_top": False, "weights": cfg.weights}
    apps = keras.applications
    if cfg.backbone == "mobilenet_v1":
        return apps.MobileNet(alpha=cfg.alpha, **common)
    if cfg.backbone == "mobilenet_v2":
        return apps.MobileNetV2(alpha=cfg.alpha, **common)
    if cfg.backbone == "mobilenet_v3_small":
        # include_preprocessing: the model rescales [0, 255] itself
        return apps.MobileNetV3Small(alpha=cfg.alpha, include_preprocessing=True, **common)
    if cfg.backbone == "efficientnet_b0":
        return apps.EfficientNetB0(**common)  # rescales [0, 255] internally
    raise ValueError(f"unknown backbone {cfg.backbone!r}")


# preprocessing each backbone's ImageNet weights expect (see leafdx.train.data)
BACKBONE_PREPROCESSING = {
    "mobilenet_v1": "mobilenet",
    "mobilenet_v2": "mobilenet",
    "mobilenet_v3_small": "raw",
    "efficientnet_b0": "raw",
}


def build_transfer(
    cfg: ModelCfg, image_size: int, stem: Stem = None, name: str | None = None
) -> keras.Model:
    base = _backbone(cfg, image_size)
    base.trainable = not cfg.freeze_backbone
    inputs, x = _input_and_stem(image_size, stem)
    # training=False keeps BatchNorm in inference mode when the base is frozen (Keras guidance
    # for transfer learning); when fine-tuning the base, BN statistics are updated.
    x = base(x, training=False if cfg.freeze_backbone else None)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(cfg.dropout, name="dropout")(x)
    outputs = layers.Dense(
        cfg.num_classes, activation=cfg.output_activation, dtype="float32", name="classifier"
    )(x)
    return keras.Model(inputs, outputs, name=name or cfg.backbone)


def build_paper_mobilenet(cfg: ModelCfg, image_size: int, stem: Stem = None) -> keras.Model:
    if cfg.backbone != "mobilenet_v1":
        raise ValueError("paper_mobilenet is MobileNet V1; use model.name=transfer otherwise")
    return build_transfer(cfg, image_size, stem, name="paper_mobilenet")


BUILDERS: dict[str, Callable[..., keras.Model]] = {
    "paper_cnn": build_paper_cnn,
    "paper_mobilenet": build_paper_mobilenet,
    "transfer": build_transfer,
}


def build_model(cfg: ModelCfg, image_size: int, stem: Stem = None) -> keras.Model:
    try:
        builder = BUILDERS[cfg.name]
    except KeyError as e:
        raise ValueError(f"unknown model {cfg.name!r}; choose from {sorted(BUILDERS)}") from e
    return builder(cfg, image_size, stem)
