"""Leakage metrics between an evaluation split and the training split.

For each evaluation image we ask whether the training split contains:
- the same **source image** (an offline-augmented variant of the same PlantVillage photo),
- the same **physical leaf** (upstream leaf-map; different photos of one leaf),
- a **byte-identical** file (SHA-256),
- a **perceptual near-duplicate** (D4 pHash distance <= t), for several thresholds t.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from leafdx.data.filenames import ORIGINAL
from leafdx.data.neighbors import pairwise_d4


def _frac(mask: pd.Series | np.ndarray) -> float:
    m = np.asarray(mask)
    return float(m.mean()) if len(m) else 0.0


def split_labels(df: pd.DataFrame, split_col: str) -> np.ndarray:
    """Split names as a plain string array; rows outside every split become ""."""
    return df[split_col].astype("string").fillna("").to_numpy(dtype=object)


def key_overlap(
    df: pd.DataFrame, split_col: str, eval_split: str, train_split: str, key: str
) -> pd.Series:
    """Boolean per eval row: does its ``key`` value also occur in the train split?"""
    split = split_labels(df, split_col)
    train_keys = set(df.loc[split == train_split, key].dropna())
    ev = df.loc[split == eval_split, key]
    return ev.notna() & ev.isin(train_keys)


def phash_overlap(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    split_col: str,
    eval_split: str,
    train_split: str,
    thresholds: Sequence[int],
) -> dict[int, pd.Series]:
    """Per threshold: boolean per eval row, has a train neighbour within distance t."""
    split = split_labels(df, split_col)
    i, j, d = (pairs[c].to_numpy() for c in ("i", "j", "dist"))
    # orient each pair as (eval, train)
    e1 = (split[i] == eval_split) & (split[j] == train_split)
    e2 = (split[j] == eval_split) & (split[i] == train_split)
    ev_idx = np.concatenate([i[e1], j[e2]])
    dist = np.concatenate([d[e1], d[e2]])
    nearest = pd.Series(dist).groupby(ev_idx).min()
    eval_rows = df.index[split == eval_split]
    nn = nearest.reindex(eval_rows)
    return {t: nn.le(t).fillna(False) for t in thresholds}


def leakage_report(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    split_col: str,
    eval_split: str,
    train_split: str,
    thresholds: Sequence[int],
) -> dict[str, Any]:
    """Summary dict plus a per-class table for one (eval, train) split pair.

    ``df`` must have a RangeIndex aligned with the indices in ``pairs`` and columns
    label, source_id, leaf_id, sha256.
    """
    ev = df[split_labels(df, split_col) == eval_split]
    src = key_overlap(df, split_col, eval_split, train_split, "source_id")
    leaf = key_overlap(df, split_col, eval_split, train_split, "leaf_id")
    sha = key_overlap(df, split_col, eval_split, train_split, "sha256")
    ph = phash_overlap(df, pairs, split_col, eval_split, train_split, thresholds)
    any_known = src | leaf | sha
    summary: dict[str, Any] = {
        "eval_split": eval_split,
        "train_split": train_split,
        "n_eval": len(ev),
        "n_eval_with_leaf_id": int(ev["leaf_id"].notna().sum()),
        "frac_source_in_train": _frac(src),
        "frac_leaf_in_train": _frac(leaf),
        "frac_leaf_in_train_among_known": _frac(leaf[ev["leaf_id"].notna()]),
        "frac_exact_duplicate_in_train": _frac(sha),
        "frac_source_or_leaf_or_exact": _frac(any_known),
        "phash": {
            str(t): {
                "frac_near_duplicate_in_train": _frac(ph[t]),
                "frac_near_duplicate_not_explained": _frac(ph[t] & ~any_known),
            }
            for t in thresholds
        },
    }
    per_class = (
        pd.DataFrame({"label": ev["label"], "source": src, "leaf": leaf, "any": any_known})
        .groupby("label")
        .agg(
            n_eval=("source", "size"),
            frac_source=("source", "mean"),
            frac_leaf=("leaf", "mean"),
            frac_any=("any", "mean"),
        )
        .reset_index()
    )
    return {"summary": summary, "per_class": per_class}


def phash_recall_on_known_variants(
    df: pd.DataFrame, hashes: np.ndarray, thresholds: Sequence[int]
) -> pd.DataFrame:
    """How often pHash detects (original, augmented variant) pairs of the same source image.

    This calibrates pHash as a leakage detector: e.g. D4 hashing should find 90-degree rotations
    and flips, but not 30-degree rotations. Returns one row per augmentation type.
    """
    orig = df[df["aug_type"] == ORIGINAL]
    orig_idx = pd.Series(orig.index, index=orig["source_id"])
    orig_idx = orig_idx[~orig_idx.index.duplicated()]
    var = df[(df["aug_type"] != ORIGINAL) & df["source_id"].isin(orig_idx.index)]
    if var.empty:
        return pd.DataFrame(columns=["aug_type", "n_pairs", "median_dist"])
    d = pairwise_d4(hashes, orig_idx.loc[var["source_id"]].to_numpy(), var.index.to_numpy())
    t = pd.DataFrame({"aug_type": var["aug_type"].to_numpy(), "dist": d.astype(int)})
    out = t.groupby("aug_type").agg(n_pairs=("dist", "size"), median_dist=("dist", "median"))
    for thr in thresholds:
        out[f"recall_at_{thr}"] = t.groupby("aug_type")["dist"].apply(
            lambda s, k=thr: (s <= k).mean()
        )
    return out.reset_index()
