"""Phase 6 methods: test-time augmentation, AdaBN and fine-tuning on field images."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from leafdx.field.data import SIZE, ImageSet


def load_model(path: str) -> keras.Model:
    """A Phase 3 inference model: raw [0, 255] RGB in, 38 softmax probabilities out."""
    return keras.models.load_model(path, compile=False)


def all_layers(model: keras.Layer) -> Iterator[keras.Layer]:
    for layer in getattr(model, "layers", []):
        yield layer
        yield from all_layers(layer)


def batchnorm_layers(model: keras.Model) -> list[keras.layers.BatchNormalization]:
    return [
        layer for layer in all_layers(model) if isinstance(layer, keras.layers.BatchNormalization)
    ]


# ------------------------------------------------------------------------------- prediction


def _views(x: tf.Tensor) -> list[tf.Tensor]:
    """Original, horizontal flip, 87.5% centre crop (resized back), and its flip."""
    crop = tf.image.resize(tf.image.central_crop(x, 0.875), (SIZE, SIZE), method="bilinear")
    return [x, tf.image.flip_left_right(x), crop, tf.image.flip_left_right(crop)]


def predict(model: keras.Model, s: ImageSet, tta: bool = False, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(s.x), batch):
        x = tf.convert_to_tensor(s.x[i : i + batch], tf.float32)
        views = _views(x) if tta else [x]
        p = np.mean([np.asarray(model(v, training=False)) for v in views], axis=0)
        out.append(p)
    return np.concatenate(out).astype(np.float32)


# ----------------------------------------------------------------------------------- AdaBN


def adabn(model: keras.Model, unlabelled: ImageSet, batch: int = 32, seed: int = 0) -> int:
    """AdaBN (Li et al., 2016): replace every batch-normalisation layer's moving mean and
    variance by statistics of the target images. One pass in training mode, with the momentum
    set so that the moving statistics become the exact average over all batches. Weights are
    not changed and labels are not used. Returns the number of BN layers adapted."""
    bns = batchnorm_layers(model)
    for bn in bns:
        bn.trainable = True
    order = np.random.default_rng(seed).permutation(len(unlabelled.x))
    for t, i in enumerate(range(0, len(order), batch)):
        for bn in bns:
            bn.momentum = t / (t + 1)  # moving = moving * t/(t+1) + batch / (t+1)
        x = tf.convert_to_tensor(unlabelled.x[order[i : i + batch]], tf.float32)
        model(x, training=True)
    for bn in bns:
        bn.momentum = 0.99
    return len(bns)


# ------------------------------------------------------------------------------ fine-tuning


def _augment(x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Random resized crop (60-100% of the area), flips and colour jitter, on [0, 255]."""
    s = tf.random.experimental.stateless_split(seed, 7)
    begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
        tf.shape(x),
        bounding_boxes=tf.constant([[[0.0, 0.0, 1.0, 1.0]]]),
        seed=s[0],
        area_range=(0.6, 1.0),
        aspect_ratio_range=(3 / 4, 4 / 3),
        use_image_if_no_bounding_boxes=True,
    )
    x = tf.image.resize(tf.slice(x, begin, size), (SIZE, SIZE), method="bilinear")
    x = tf.image.stateless_random_flip_left_right(x, s[1])
    x = tf.image.stateless_random_flip_up_down(x, s[2])
    # colour ops in [0, 1], the range tf.image expects for saturation
    x = x / 255.0
    x = tf.image.stateless_random_brightness(x, 0.2, s[3])
    x = tf.image.stateless_random_contrast(x, 0.8, 1.2, s[4])
    x = tf.image.stateless_random_saturation(x, 0.8, 1.2, s[5])
    return tf.clip_by_value(x, 0.0, 1.0) * 255.0


def _train_dataset(x: np.ndarray, y: np.ndarray, batch: int, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=seed)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b))
    rng = tf.data.Dataset.random(seed=seed).batch(2)
    ds = tf.data.Dataset.zip((ds, rng)).map(
        lambda ab, r: (_augment(ab[0], tf.cast(r, tf.int32)), ab[1]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def finetune(
    model: keras.Model,
    train: list[ImageSet],
    val: ImageSet,
    lr: float,
    epochs: int,
    patience: int,
    seed: int,
    batch: int = 32,
) -> dict[str, Any]:
    """Fine-tune the whole network on the union of ``train`` sets, with batch-normalisation
    layers in inference mode, early stopping on ``val`` loss and the best weights restored."""
    keras.utils.set_random_seed(seed)
    for bn in batchnorm_layers(model):
        bn.trainable = False  # keep the moving statistics; with Keras 3 this is inference mode
    x = np.concatenate([s.x for s in train])
    y = np.concatenate([s.y for s in train])
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    hist = model.fit(
        _train_dataset(x, y, batch, seed),
        validation_data=(val.x.astype(np.float32), val.y),
        epochs=epochs,
        callbacks=[stop],
        verbose=2,
    )
    h = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    best = int(np.argmin(h["val_loss"]))
    return {"n_train": len(x), "epochs_run": len(h["loss"]), "best_epoch": best + 1, "history": h}
