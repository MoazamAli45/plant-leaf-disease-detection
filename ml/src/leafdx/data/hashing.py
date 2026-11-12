"""Image indexing: checksums, perceptual hashes and cheap image statistics.

The perceptual hash is the standard DCT pHash (Zauner 2010; identical to ``imagehash.phash``
for the identity transform). Because NPD's offline augmentation rotates by multiples of 90
degrees and mirrors, each image gets **eight** hashes, one per element of the dihedral group
D4 (4 rotations x optional mirror), computed on the 32x32 grayscale thumbnail. Comparing the
identity hash of A against all eight hashes of B makes the distance invariant to those
transforms (see :mod:`leafdx.data.neighbors`). Resizing commutes with 90-degree rotation and
mirroring, so transforming the thumbnail is equivalent to transforming the image.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct
from tqdm import tqdm

log = logging.getLogger(__name__)

N_TRANSFORMS = 8
HASH_COLS = [f"phash{k}" for k in range(N_TRANSFORMS)]
STAT_THUMB = 64
BORDER = 4  # px of the 64x64 thumbnail treated as "border" (background proxy)


def _phash_bits(pixels: np.ndarray, hash_size: int = 8) -> int:
    coeffs = dct(dct(pixels, axis=0), axis=1)
    low = coeffs[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def d4_transforms(a: np.ndarray) -> list[np.ndarray]:
    """The 8 dihedral transforms; index 0 is the identity."""
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out.extend([r, np.fliplr(r)])
    return out


def phash_d4(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> list[int]:
    """Eight 64-bit pHashes of ``img`` (identity first)."""
    side = hash_size * highfreq_factor
    thumb = img.convert("L").resize((side, side), Image.Resampling.LANCZOS)
    pixels = np.asarray(thumb, dtype=np.float64)
    return [_phash_bits(t, hash_size) for t in d4_transforms(pixels)]


def image_stats(img: Image.Image) -> dict[str, float]:
    """Mean RGB, grayscale std and border/centre brightness on a 64x64 thumbnail."""
    rgb = np.asarray(
        img.convert("RGB").resize((STAT_THUMB, STAT_THUMB), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    gray = rgb.mean(axis=2)
    mask = np.zeros(gray.shape, dtype=bool)
    mask[:BORDER, :] = mask[-BORDER:, :] = mask[:, :BORDER] = mask[:, -BORDER:] = True
    return {
        "mean_r": float(rgb[..., 0].mean()),
        "mean_g": float(rgb[..., 1].mean()),
        "mean_b": float(rgb[..., 2].mean()),
        "std_gray": float(gray.std()),
        "border_mean": float(gray[mask].mean()),
        "centre_mean": float(gray[~mask].mean()),
    }


def index_image(path: Path) -> dict[str, Any]:
    """Everything we need to know about one file. Never raises; failures go in ``error``."""
    rec: dict[str, Any] = {"error": ""}
    try:
        data = path.read_bytes()
        rec["sha256"] = hashlib.sha256(data).hexdigest()
        rec["bytes"] = len(data)
        with Image.open(path) as img:
            img.load()
            rec.update(width=img.width, height=img.height, mode=img.mode, format=img.format)
            hashes = phash_d4(img)
            rec.update({c: f"{h:016x}" for c, h in zip(HASH_COLS, hashes, strict=True)})
            rec.update(image_stats(img))
    except Exception as e:  # corrupt files are data, not crashes
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def _index_one(args: tuple[str, str]) -> dict[str, Any]:
    root, rel = args
    return {"path": rel, **index_image(Path(root) / rel)}


def index_images(
    root: Path,
    rel_paths: Sequence[str],
    cache: Path | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    """Index ``rel_paths`` under ``root`` in parallel, reusing a CSV cache keyed by path+size."""
    cached = pd.DataFrame()
    if cache is not None and cache.exists():
        cached = pd.read_csv(cache, dtype={c: str for c in HASH_COLS}, keep_default_na=False)
        sizes = {p: (root / p).stat().st_size for p in rel_paths if (root / p).exists()}
        ok = cached["path"].map(sizes).eq(cached["bytes"]) & cached["path"].isin(sizes)
        cached = cached[ok]
    todo = sorted(set(rel_paths) - set(cached.get("path", pd.Series(dtype=str))))
    log.info("indexing %d images (%d cached)", len(todo), len(rel_paths) - len(todo))
    rows: list[dict[str, Any]] = []
    if todo:
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            it = ex.map(_index_one, [(str(root), p) for p in todo], chunksize=64)
            rows = list(tqdm(it, total=len(todo), desc="index", unit="img"))
    df = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
    df = df[df["path"].isin(set(rel_paths))].sort_values("path").reset_index(drop=True)
    if cache is not None and rows:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def hashes_to_array(df: pd.DataFrame) -> np.ndarray:
    """(n, 8) uint64 array from the hex hash columns."""
    return np.array(
        [[int(h, 16) for h in row] for row in df[HASH_COLS].itertuples(index=False)],
        dtype=np.uint64,
    ).reshape(len(df), N_TRANSFORMS)
