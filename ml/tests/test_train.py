"""Training stack: models, config, augmentation, metrics and a smoke run (CPU, tiny data)."""

import json

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")
from PIL import Image  # noqa: E402

from leafdx.data.classes import PLANTVILLAGE_CLASSES  # noqa: E402
from leafdx.train.config import ModelCfg, load_config  # noqa: E402
from leafdx.train.data import (  # noqa: E402
    RandomBrightnessMultiplicative,
    build_augmentation,
    resolve_paths,
)
from leafdx.train.metrics import classification_metrics  # noqa: E402
from leafdx.train.models import build_paper_cnn, build_paper_mobilenet  # noqa: E402


def test_paper_cnn_matches_figure_2():
    m = build_paper_cnn(ModelCfg(), 224)
    # hand count: convs 896 + 9248 + 9248 + 36992; flatten 12*12*128 = 18432;
    # dense 18432*128+128, 128*64+64, 64*38+38
    assert m.count_params() == 2_426_534
    assert m.layers[-1].activation.__name__ == "sigmoid"
    assert [type(layer).__name__ for layer in m.layers].count("Dropout") == 3


def test_paper_mobilenet_frozen_head_only():
    m = build_paper_mobilenet(
        ModelCfg(name="paper_mobilenet", output_activation="softmax", weights=None), 224
    )
    trainable = sum(int(np.prod(w.shape)) for w in m.trainable_weights)
    assert trainable == 1024 * 38 + 38
    assert m.count_params() == 3_228_864 + trainable


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\ntrain:\n  epochs: 20\naugment:\n  brightness_range: [0.5, 1.5]\n")
    c = load_config(p, ["train.epochs=3", "seed=7", "train.steps_per_epoch=null"])
    assert (c.train.epochs, c.seed, c.train.steps_per_epoch) == (3, 7, None)
    assert c.augment.brightness_range == (0.5, 1.5)
    with pytest.raises(KeyError):
        load_config(p, ["train.epoch=3"])


def test_resolve_paths_npd_root():
    from leafdx.train.config import DataCfg

    cfg = DataCfg(data_root="data", npd_root="/kaggle/input/npd")
    assert resolve_paths(["raw/npd/train/A/x.jpg", "raw/plantdoc/t/a.jpg"], cfg) == [
        "/kaggle/input/npd/train/A/x.jpg",
        "data/raw/plantdoc/t/a.jpg",
    ]


def test_augmentation_keeps_shape_and_range():
    from leafdx.train.config import AugmentCfg

    aug = build_augmentation(AugmentCfg(), seed=0)
    x = tf.random.uniform((4, 64, 64, 3), 0, 255)
    y = aug(x, training=True)
    assert y.shape == x.shape
    assert float(tf.reduce_min(y)) >= 0 and float(tf.reduce_max(y)) <= 255.0 + 1e-3
    np.testing.assert_allclose(aug(x, training=False), x)


def test_brightness_is_multiplicative_and_clipped():
    layer = RandomBrightnessMultiplicative(2.0, 2.0)
    x = tf.fill((1, 2, 2, 3), 100.0)
    np.testing.assert_allclose(layer(x, training=True), 200.0)
    np.testing.assert_allclose(layer(tf.fill((1, 2, 2, 3), 200.0), training=True), 255.0)


def test_classification_metrics():
    y = np.array([0, 0, 1, 2])
    probs = np.eye(38)[[0, 1, 1, 2]].astype(float)
    summary, per_class, cm = classification_metrics(y, probs, PLANTVILLAGE_CLASSES)
    assert summary["accuracy"] == 0.75
    assert per_class.loc[1, "precision"] == 0.5
    assert cm[0, 1] == 1 and cm.shape == (38, 38)


@pytest.mark.parametrize("cfg_name", ["cnn_steps100", "mobilenet_fullepochs"])
def test_smoke_run(tmp_path, cfg_name):
    from pathlib import Path

    from leafdx.train.run import run

    rng = np.random.default_rng(0)
    rows = {"train": [], "val": []}
    for split, n in (("train", 2), ("val", 1)):
        for c, cls in enumerate(PLANTVILLAGE_CLASSES):
            d = tmp_path / "data" / "raw" / "npd" / ("valid" if split == "val" else "train") / cls
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                f = d / f"{c}_{k}.jpg"
                Image.fromarray(rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)).save(f)
                rows[split].append(
                    {"path": f.relative_to(tmp_path / "data").as_posix(), "label": cls}
                )
    for split, r in rows.items():
        pd.DataFrame(r).to_csv(tmp_path / f"{split}.csv", index=False)

    cfg_path = Path(__file__).parents[1] / "configs" / "paper" / f"{cfg_name}.yaml"
    over = [
        f"data.train_manifest={tmp_path / 'train.csv'}",
        f"data.val_manifest={tmp_path / 'val.csv'}",
        f"data.test_manifest={tmp_path / 'val.csv'}",
        f"data.data_root={tmp_path / 'data'}",
        "data.image_size=64",
        f"output_dir={tmp_path / 'out'}",
        "train.epochs=1",
        "train.batch_size=8",
        "model.weights=null",
    ]
    if "steps100" in cfg_name:
        over += ["train.steps_per_epoch=2", "train.validation_steps=2"]
    m = run(load_config(cfg_path, over))
    run_dir = tmp_path / "out" / m["name"] / "seed42"
    for f in [
        "config.yaml",
        "env.json",
        "history.csv",
        "metrics.json",
        "model.keras",
        "per_class_val.csv",
        "probs_test.npz",
    ]:
        assert (run_dir / f).exists(), f
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved["val_full"]["n"] == len(PLANTVILLAGE_CLASSES)
    assert {"accuracy", "loss", "val_accuracy", "val_loss"} <= set(saved["last_epoch"])
    if "cnn" in cfg_name:
        assert "binary_accuracy" in saved["last_epoch"]


# ---------------------------------------------------------------- Phase 3


@pytest.mark.parametrize(
    ("backbone", "prep"),
    [
        ("mobilenet_v1", "mobilenet"),
        ("mobilenet_v2", "mobilenet"),
        ("mobilenet_v3_small", "raw"),
        ("efficientnet_b0", "raw"),
    ],
)
def test_transfer_backbones_build_with_stem(backbone, prep):
    from leafdx.train.config import AugmentCfg
    from leafdx.train.data import preprocessing_layer
    from leafdx.train.models import build_model

    cfg = ModelCfg(
        name="transfer",
        backbone=backbone,
        output_activation="softmax",
        weights=None,
        freeze_backbone=False,
    )
    stem = [build_augmentation(AugmentCfg(), 0, dtype="float32"), preprocessing_layer(prep)]
    m = build_model(cfg, 96, stem=stem)
    out = m(tf.random.uniform((2, 96, 96, 3), 0, 255), training=False)
    assert out.shape == (2, 38)
    np.testing.assert_allclose(out.numpy().sum(axis=1), 1.0, rtol=1e-4)


def test_model_with_stem_roundtrips(tmp_path):
    import keras

    from leafdx.train.config import AugmentCfg
    from leafdx.train.data import preprocessing_layer
    from leafdx.train.models import build_model

    stem = [build_augmentation(AugmentCfg(), 0, dtype="float32"), preprocessing_layer("rescale")]
    m = build_model(ModelCfg(output_activation="softmax"), 64, stem=stem)
    x = tf.random.uniform((3, 64, 64, 3), 0, 255)
    m.save(tmp_path / "m.keras")
    m2 = keras.models.load_model(tmp_path / "m.keras")
    # augmentation is inactive at inference, so both give the same output
    np.testing.assert_allclose(m(x, training=False), m2(x, training=False), atol=1e-6)


def test_restricted_accuracy_and_macro_over_present():
    y = np.array([0, 1])  # only classes 0 and 1 present
    probs = np.zeros((2, 38))
    probs[0, [0, 5]] = [0.4, 0.6]  # wrong overall (5), right among present classes
    probs[1, 1] = 1.0
    s, _, _ = classification_metrics(y, probs, PLANTVILLAGE_CLASSES)
    assert s["accuracy"] == 0.5 and s["restricted_accuracy"] == 1.0
    assert s["n_classes_present"] == 2
    assert s["macro_recall"] == 0.5  # recall: class 0 -> 0, class 1 -> 1; mean over present


def test_ece_perfect_and_overconfident():
    from leafdx.train.metrics import expected_calibration_error

    y = np.array([0, 1, 2, 3])
    perfect = np.eye(38)[y]
    assert expected_calibration_error(y, perfect) == 0.0
    wrong = np.eye(38)[[1, 2, 3, 0]]  # 100% confident, always wrong
    assert expected_calibration_error(y, wrong) == pytest.approx(1.0)


def test_phase3_smoke_run(tmp_path):
    from pathlib import Path

    from leafdx.train.run import run

    rng = np.random.default_rng(1)
    rows = {"train": [], "val": [], "test": [], "pd": []}
    for c, cls in enumerate(PLANTVILLAGE_CLASSES):
        d = tmp_path / "data" / "raw" / "npd" / "train" / cls
        d.mkdir(parents=True)
        for k in range(4):
            f = d / f"{c}_{k}.jpg"
            Image.fromarray(rng.integers(0, 255, (72, 72, 3), dtype=np.uint8)).save(f)
            rel = f.relative_to(tmp_path / "data").as_posix()
            split = ["train", "train", "val", "test"][k]
            rows[split].append(
                {"path": rel, "label": cls, "aug_type": "original" if c % 2 else "90deg"}
            )
    for c, cls in enumerate(PLANTVILLAGE_CLASSES[:5]):  # field set: 5 classes only
        d = tmp_path / "data" / "raw" / "plantdoc" / "test" / f"pd{c}"
        d.mkdir(parents=True)
        f = d / "a.jpg"
        Image.fromarray(rng.integers(0, 255, (90, 60, 3), dtype=np.uint8)).save(f)
        rows["pd"].append(
            {
                "path": f.relative_to(tmp_path / "data").as_posix(),
                "label": cls,
                "match": "exact" if c < 3 else "approximate",
            }
        )
    for k, r in rows.items():
        pd.DataFrame(r).to_csv(tmp_path / f"{k}.csv", index=False)

    cfg_path = Path(__file__).parents[1] / "configs" / "phase3" / "mobilenet_v3_small.yaml"
    over = [
        f"data.train_manifest={tmp_path / 'train.csv'}",
        f"data.val_manifest={tmp_path / 'val.csv'}",
        f"data.data_root={tmp_path / 'data'}",
        "data.image_size=64",
        f"output_dir={tmp_path / 'out'}",
        "train.epochs=2",
        "train.batch_size=16",
        "train.mixed_precision=false",
        "model.weights=null",
        f"eval=[{{name: test, manifest: {tmp_path / 'test.csv'},"
        " subsets: {original: \"aug_type == 'original'\"}},"
        f" {{name: plantdoc, manifest: {tmp_path / 'pd.csv'},"
        " subsets: {exact: \"match == 'exact'\"}}]",
    ]
    m = run(load_config(cfg_path, over))
    run_dir = tmp_path / "out" / "p3_mobilenet_v3_small" / "seed42"
    assert (run_dir / "model.keras").exists() and (run_dir / "probs_plantdoc.npz").exists()
    assert m["eval"]["test"]["n"] == 38
    assert m["eval"]["test"]["subsets"]["original"]["n"] == 19
    assert m["eval"]["plantdoc"]["n_classes_present"] == 5
    assert m["eval"]["plantdoc"]["subsets"]["exact"]["n"] == 3
    assert 1 <= m["best_epoch"] <= m["epochs_run"] == 2
