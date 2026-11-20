"""Phase 6 (field adaptation) building blocks."""

from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from leafdx.field.data import ImageSet, k_shot
from leafdx.field.methods import _augment, adabn, batchnorm_layers, predict

MANIFESTS = Path(__file__).resolve().parents[1] / "manifests" / "field_split"


def _set(n_per_class: int, classes: int = 3) -> ImageSet:
    y = np.repeat(np.arange(classes), n_per_class)
    x = np.zeros((len(y), 224, 224, 3), np.uint8)
    return ImageSet("t", x, y, [f"img{i}" for i in range(len(y))])


def test_k_shot_takes_at_most_k_per_class_and_depends_on_seed():
    s = _set(10)
    a, b, c = k_shot(s, 4, 0), k_shot(s, 4, 0), k_shot(s, 4, 1)
    assert np.bincount(a.y).tolist() == [4, 4, 4]
    assert a.paths == b.paths
    assert a.paths != c.paths
    assert k_shot(s, None, 0) is s
    assert np.bincount(k_shot(_set(2), 5, 0).y).tolist() == [2, 2, 2]  # fewer than k available


def test_augment_is_seeded_and_stays_in_range():
    x = tf.constant(np.random.default_rng(0).integers(0, 256, (224, 224, 3)), tf.float32)
    a = _augment(x, tf.constant([1, 2]))
    assert a.shape == (224, 224, 3)
    assert bool(tf.reduce_all(a == _augment(x, tf.constant([1, 2]))))
    assert not bool(tf.reduce_all(a == _augment(x, tf.constant([3, 4]))))
    assert float(tf.reduce_min(a)) >= 0 and float(tf.reduce_max(a)) <= 255


def _bn_model() -> keras.Model:
    inp = keras.Input((224, 224, 3))
    h = keras.layers.GlobalAveragePooling2D()(inp)
    h = keras.layers.BatchNormalization(name="bn")(h)
    out = keras.layers.Dense(4, activation="softmax")(h)
    return keras.Model(inp, out)


def test_adabn_sets_moving_stats_to_the_target_average():
    model = _bn_model()
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (64, 224, 224, 3)).astype(np.uint8)
    target = ImageSet("u", x, np.zeros(64, np.int64), [str(i) for i in range(64)])
    assert adabn(model, target, batch=16) == 1
    feats = x.astype(np.float32).mean(axis=(1, 2))  # what the BN layer sees
    bn = batchnorm_layers(model)[0]
    # moving mean = average of the 4 batch means = overall mean (equal batch sizes)
    np.testing.assert_allclose(np.asarray(bn.moving_mean), feats.mean(axis=0), rtol=1e-4)
    assert bn.momentum == 0.99  # restored


def test_predict_with_tta_returns_probabilities():
    model = _bn_model()
    s = _set(2)
    for tta in (False, True):
        p = predict(model, s, tta=tta)
        assert p.shape == (6, 4)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.skipif(not (MANIFESTS / "test.csv").exists(), reason="field split not generated")
def test_committed_field_split_is_grouped_and_disjoint():
    parts = {s: pd.read_csv(MANIFESTS / f"{s}.csv") for s in ("train", "val", "test")}
    paths = [set(df["path"]) for df in parts.values()]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    both = pd.concat([df.assign(split=s) for s, df in parts.items()])
    assert both.groupby("group")["split"].nunique().max() == 1
    assert len(both) == 2557


def test_balanced_accuracy_weights_classes_equally():
    from leafdx.field.phase7 import balanced_accuracy

    y = np.array([0] * 90 + [1] * 10)
    probs = np.zeros((100, 2))
    probs[:, 0] = 1.0  # always predicts the majority class
    assert balanced_accuracy(y, probs) == 0.5  # plain accuracy would be 0.9


def test_bootstrap_interval_contains_the_estimate_and_is_seeded():
    from leafdx.field.phase7 import bootstrap_ci

    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    probs = np.eye(3)[np.where(rng.random(300) < 0.6, y, (y + 1) % 3)]
    acc = float((probs.argmax(1) == y).mean())
    lo, hi = bootstrap_ci(y, probs)
    assert lo < acc < hi and hi - lo < 0.15
    assert (lo, hi) == bootstrap_ci(y, probs)
