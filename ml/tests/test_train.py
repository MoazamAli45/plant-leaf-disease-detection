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
