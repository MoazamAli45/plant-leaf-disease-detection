"""Image sets for Phase 6, loaded once and cached as uint8 arrays (data/cache/p6/)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from leafdx.data.classes import CLASS_TO_INDEX

SIZE = 224
FIELD = Path("manifests/field_split")
CLEAN = Path("manifests/clean_split")


@dataclass(frozen=True)
class ImageSet:
    name: str
    x: np.ndarray  # (n, 224, 224, 3) uint8
    y: np.ndarray  # (n,) class indices
    paths: list[str]


def _decode(path: Path) -> np.ndarray:
    import tensorflow as tf

    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    # same resize as every earlier evaluation: bilinear straight to 224 x 224
    out = tf.image.resize(img, (SIZE, SIZE), method="bilinear")
    return np.clip(np.round(out.numpy()), 0, 255).astype(np.uint8)


def load(name: str, df: pd.DataFrame, data_root: Path, cache_dir: Path) -> ImageSet:
    """Images for the rows of ``df`` (columns ``path``, ``label``), cached under ``name``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}.npz"
    paths = df["path"].tolist()
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["paths"]) == paths:
            return ImageSet(name, z["x"], z["y"], paths)
    x = np.stack([_decode(data_root / p) for p in paths])
    y = np.array([CLASS_TO_INDEX[lab] for lab in df["label"]], dtype=np.int64)
    np.savez(cache, x=x, y=y, paths=np.array(paths))
    return ImageSet(name, x, y, paths)


def field(split: str, data_root: Path, cache_dir: Path) -> ImageSet:
    return load(f"field_{split}", pd.read_csv(FIELD / f"{split}.csv"), data_root, cache_dir)


def lab_check(data_root: Path, cache_dir: Path, per_class: int = 20, seed: int = 0) -> ImageSet:
    """The efficiency study's lab sample: up to ``per_class`` leak-free clean-test images per
    class (746 images with the defaults)."""
    df = pd.read_csv(CLEAN / "test.csv")
    links = pd.read_csv(CLEAN / "test_links_corrected.csv")
    df = df.merge(links[["path", "linked_any"]], on="path")
    df = df[~df["linked_any"]].sample(frac=1.0, random_state=seed).groupby("label").head(per_class)
    return load(f"lab_check_{per_class}_{seed}", df, data_root, cache_dir)


def lab_replay(data_root: Path, cache_dir: Path, per_class: int, seed: int) -> ImageSet:
    """Lab training images mixed into fine-tuning to limit forgetting (clean train split)."""
    df = pd.read_csv(CLEAN / "train.csv")
    df = df.sample(frac=1.0, random_state=1000 + seed).groupby("label").head(per_class)
    return load(f"lab_replay_{per_class}_{seed}", df, data_root, cache_dir)


def k_shot(s: ImageSet, k: int | None, seed: int) -> ImageSet:
    """Up to ``k`` images per class from ``s`` (all when ``k`` is None), chosen by ``seed``."""
    if k is None:
        return s
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for c in np.unique(s.y):
        idx = np.flatnonzero(s.y == c)
        keep.extend(rng.permutation(idx)[:k].tolist())
    keep.sort()
    return ImageSet(f"{s.name}_k{k}_s{seed}", s.x[keep], s.y[keep], [s.paths[i] for i in keep])
