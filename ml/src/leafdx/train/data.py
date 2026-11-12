"""tf.data input pipeline driven by the CSV manifests.

Emulates the paper's (presumed) ``ImageDataGenerator.flow_from_directory`` setup:
decode -> resize (nearest) -> augment on [0, 255] -> rescale -> one-hot labels. Removed from
Keras 3, ImageDataGenerator is replaced by preprocessing layers with its parameters mapped
one-to-one where Keras allows (see :func:`build_augmentation` and docs/reproduction.md).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from leafdx.data.classes import CLASS_TO_INDEX
from leafdx.train.config import AugmentCfg, DataCfg

NPD_PREFIX = "raw/npd/"
HAS_RANDOM_SHEAR = hasattr(keras.layers, "RandomShear")
AUTOTUNE = tf.data.AUTOTUNE


def resolve_paths(rel_paths: Sequence[str], cfg: DataCfg) -> list[str]:
    """Map manifest paths to files on this machine (``npd_root`` overrides raw/npd/)."""
    out = []
    for p in rel_paths:
        if cfg.npd_root and p.startswith(NPD_PREFIX):
            out.append(str(Path(cfg.npd_root) / p[len(NPD_PREFIX) :]))
        else:
            out.append(str(Path(cfg.data_root) / p))
    return out


def load_manifest(path: str | Path, cfg: DataCfg) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path, keep_default_na=False)
    unknown = set(df["label"]) - set(CLASS_TO_INDEX)
    if unknown:
        raise ValueError(f"{path}: unknown labels {sorted(unknown)}")
    labels = df["label"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int32)
    return resolve_paths(df["path"].tolist(), cfg), labels


class RandomBrightnessMultiplicative(keras.layers.Layer):
    """``ImageDataGenerator(brightness_range=(lo, hi))``: scale pixels by U(lo, hi), clip."""

    def __init__(self, low: float, high: float, seed: int | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.low, self.high = float(low), float(high)
        self.seed_gen = keras.random.SeedGenerator(seed)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if not training:
            return x
        batch = keras.ops.shape(x)[0]
        f = keras.random.uniform((batch, 1, 1, 1), self.low, self.high, seed=self.seed_gen)
        return keras.ops.clip(x * f, 0.0, 255.0)


def build_augmentation(cfg: AugmentCfg, seed: int) -> keras.Sequential:
    """Keras layers equivalent to the ImageDataGenerator parameters in ``cfg``.

    Mapping (ImageDataGenerator -> Keras 3):
    - rotation_range (deg)   -> RandomRotation(factor = deg / 360)
    - width/height_shift     -> RandomTranslation(same fractions)
    - shear_range (deg)      -> RandomShear(x_factor = tan(deg)); Keras samples positive shears
    - zoom_range z           -> RandomZoom(+-z height, +-z width, independent like IDG)
    - horizontal/vertical    -> RandomFlip
    - brightness_range       -> RandomBrightnessMultiplicative
    """
    aug: list[keras.layers.Layer] = []
    fill = cfg.fill_mode
    if cfg.rotation_range:
        aug.append(
            keras.layers.RandomRotation(cfg.rotation_range / 360.0, fill_mode=fill, seed=seed)
        )
    if cfg.width_shift_range or cfg.height_shift_range:
        aug.append(
            keras.layers.RandomTranslation(
                cfg.height_shift_range, cfg.width_shift_range, fill_mode=fill, seed=seed + 1
            )
        )
    if cfg.shear_range and not HAS_RANDOM_SHEAR:
        # RandomShear arrived in Keras 3.8; at IDG's 0.2 degrees the shear is ~0.35% of width
        warnings.warn("keras.layers.RandomShear unavailable: shear skipped", stacklevel=2)
    elif cfg.shear_range:
        aug.append(
            keras.layers.RandomShear(
                x_factor=math.tan(math.radians(cfg.shear_range)), fill_mode=fill, seed=seed + 2
            )
        )
    if cfg.zoom_range:
        z = (-cfg.zoom_range, cfg.zoom_range)
        aug.append(keras.layers.RandomZoom(z, z, fill_mode=fill, seed=seed + 3))
    mode = {
        (True, True): "horizontal_and_vertical",
        (True, False): "horizontal",
        (False, True): "vertical",
    }.get((cfg.horizontal_flip, cfg.vertical_flip))
    if mode:
        aug.append(keras.layers.RandomFlip(mode, seed=seed + 4))
    if cfg.brightness_range:
        aug.append(RandomBrightnessMultiplicative(*cfg.brightness_range, seed=seed + 5))
    return keras.Sequential(aug, name="augmentation")


def _preprocess_fn(kind: str) -> keras.layers.Layer:
    if kind == "rescale":
        return keras.layers.Rescaling(1.0 / 255)
    if kind == "mobilenet":
        return keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)
    raise ValueError(f"unknown preprocessing {kind!r}")


def make_dataset(
    paths: Sequence[str],
    labels: np.ndarray,
    data_cfg: DataCfg,
    batch_size: int,
    num_classes: int,
    *,
    shuffle: bool,
    repeat: bool,
    seed: int,
    augment: keras.Sequential | None = None,
    shuffle_buffer: int | None = None,
) -> tf.data.Dataset:
    size = data_cfg.image_size
    method = data_cfg.resize_method

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size), method=method)
        img.set_shape((size, size, 3))
        return tf.cast(img, tf.float32), tf.one_hot(label, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if shuffle:
        # Manifests are sorted by class, so a partial buffer would give one-class batches.
        # None = shuffle the full list of paths every epoch, like flow_from_directory.
        buf = len(paths) if shuffle_buffer is None else min(shuffle_buffer, len(paths))
        ds = ds.shuffle(buf, seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch_size)
    if augment is not None:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    prep = _preprocess_fn(data_cfg.preprocessing)
    ds = ds.map(lambda x, y: (prep(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
