"""End-to-end run of the data pipeline on a synthetic dataset with planted leaks."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageOps

from leafdx.data.classes import PLANTVILLAGE_CLASSES
from leafdx.data.prepare import DataConfig, run


def textured(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (10, 10, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((64, 64), Image.Resampling.BICUBIC)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    data = tmp / "data"
    npd = data / "raw" / "npd"
    leaf_map: dict[str, list[str]] = {}
    seed = 0
    for c, cls in enumerate(PLANTVILLAGE_CLASSES):
        for split in ("train", "valid"):
            (npd / split / cls).mkdir(parents=True)
        imgs = {}
        for k in range(4):
            seed += 1
            imgs[k] = textured(seed)
            stem = f"u{c}k{k}___ID {c * 10 + k}"
            folder = "valid" if k == 2 else "train"
            imgs[k].save(npd / folder / cls / f"{stem}.png")
        # s0: a 90-degree copy ends up in valid -> source-level leak
        imgs[0].rotate(90, expand=True).save(
            npd / "valid" / cls / f"u{c}k0___ID {c * 10}_90deg.png"
        )
        # s1: a mirrored copy stays in train
        ImageOps.mirror(imgs[1]).save(npd / "train" / cls / f"u{c}k1___ID {c * 10 + 1}_flipLR.png")
        # s2 (valid) and s3 (train) are photos of the same physical leaf in class 0 only
        if c == 0:
            leaf_map[f"id {c * 10 + 2}"] = [f"{cls}:::1.0"]
            leaf_map[f"id {c * 10 + 3}"] = [f"{cls}:::1.0"]
    (data / "raw" / "plantvillage").mkdir(parents=True)
    (data / "raw" / "plantvillage" / "leaf-map.json").write_text(json.dumps(leaf_map))

    pdoc = data / "raw" / "plantdoc"
    (pdoc / "train" / "Apple leaf").mkdir(parents=True)
    (pdoc / "test" / "Tomato leaf").mkdir(parents=True)
    for k in range(3):
        textured(10_000 + k).save(pdoc / "train" / "Apple leaf" / f"f{k}.png")
    textured(20_000).save(pdoc / "test" / "Tomato leaf" / "t.png")
    # a PlantDoc image that is really an NPD image -> must be excluded from the field test set
    src = next((npd / "train" / PLANTVILLAGE_CLASSES[3]).glob("*k3*"))
    shutil.copy(src, pdoc / "train" / "Apple leaf" / "copied.png")

    cfg = DataConfig(
        data_dir=data, manifest_dir=tmp / "manifests", report_dir=tmp / "reports", workers=2
    )
    summary = run(cfg)
    return tmp, summary


def test_paper_split_leakage_is_measured(result):
    _, s = result
    a = s["paper_split_audit"]
    n_classes = len(PLANTVILLAGE_CLASSES)
    assert a["n_eval"] == 2 * n_classes
    assert a["frac_source_in_train"] == pytest.approx(0.5)
    assert a["frac_leaf_in_train"] == pytest.approx(1 / (2 * n_classes))
    # the 90-degree copy is found by D4 pHash at a small threshold
    assert a["phash"]["4"]["frac_near_duplicate_in_train"] >= 0.5


def test_clean_split_has_no_leakage(result):
    _, s = result
    for audit in s["clean_split_audit"].values():
        assert audit["frac_source_or_leaf_or_exact"] == 0
        assert audit["phash"]["4"]["frac_near_duplicate_in_train"] == 0


def test_groups_respected_in_manifests(result):
    tmp, _ = result
    clean = pd.concat(
        [
            pd.read_csv(tmp / "manifests" / "clean_split" / f"{s}.csv").assign(split=s)
            for s in ("train", "val", "test")
        ]
    )
    assert clean.groupby("group_id")["split"].nunique().max() == 1
    assert clean.groupby("source_id")["split"].nunique().max() == 1
    assert len(clean) == 6 * len(PLANTVILLAGE_CLASSES)


def test_plantdoc_copy_excluded(result):
    tmp, s = result
    assert s["plantdoc"]["n_images"] == 5
    assert s["plantdoc"]["n_test"] == 4
    allpd = pd.read_csv(tmp / "manifests" / "plantdoc" / "all.csv", keep_default_na=False)
    excluded = allpd[allpd["exclude_reason"] != ""]
    assert excluded["path"].str.endswith("copied.png").all() and len(excluded) == 1


def test_checksums_written(result):
    tmp, _ = result
    lines = (tmp / "manifests" / "SHA256SUMS").read_text().splitlines()
    assert {Path(line.split("  ")[1]).name for line in lines} >= {"train.csv", "test.csv"}


def test_recall_table(result):
    tmp, _ = result
    r = pd.read_csv(tmp / "reports" / "phash_recall_by_aug_type.csv")
    assert set(r["aug_type"]) == {"90deg", "flipLR"}
    assert (r["recall_at_4"] == 1.0).all()
