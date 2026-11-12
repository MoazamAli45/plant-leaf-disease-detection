"""Grouping and grouped, stratified splitting.

Images are linked when they share a source image, a physical leaf, or are near-duplicates.
Connected components of that graph are *groups*; a group is never divided between splits.
Groups are then assigned split by split, per class, so that each class has (close to) the
target proportions in train/val/test.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def connected_groups(n: int, edges: Iterable[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Component id (0..G-1) per node, given edge index arrays (i, j)."""
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for i, j in edges:
        rows.append(np.asarray(i, dtype=np.int64))
        cols.append(np.asarray(j, dtype=np.int64))
    r = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    return comp.astype(np.int64)


def edges_from_keys(keys: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    """Star edges linking all rows that share a non-null key."""
    s = pd.Series(list(keys), dtype=object)
    valid = (s.notna() & (s.astype(str) != "")).to_numpy()
    idx = np.flatnonzero(valid)
    codes, _ = pd.factorize(s[valid])
    _, first = np.unique(codes, return_index=True)
    i = idx[first[codes]]
    keep = i != idx
    return i[keep].astype(np.int64), idx[keep].astype(np.int64)


def stratified_group_split(
    labels: Sequence[str],
    groups: Sequence[int],
    ratios: Mapping[str, float],
    seed: int,
) -> np.ndarray:
    """Assign each row to a split name, keeping groups intact and classes stratified.

    Each group takes the majority label of its members. Within each class, groups are visited
    largest first (random order among equal sizes) and given to the split furthest below its
    target count. Deterministic for a given seed.
    """
    total = sum(ratios.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"ratios must sum to 1, got {total}")
    names = list(ratios)
    df = pd.DataFrame({"label": np.asarray(labels), "group": np.asarray(groups)})
    g = df.groupby("group")["label"].agg(lambda s: s.value_counts().sort_index().idxmax())
    sizes = df.groupby("group").size()
    table = pd.DataFrame({"label": g, "size": sizes})

    rng = np.random.default_rng(seed)
    assign: dict[int, str] = {}
    for label in sorted(table["label"].unique()):
        sub = table[table["label"] == label]
        order = rng.permutation(len(sub))
        sub = sub.iloc[order]
        sub = sub.iloc[np.argsort(-sub["size"].to_numpy(), kind="stable")]
        n_label = int(sub["size"].sum())
        target = {s: ratios[s] * n_label for s in names}
        current = dict.fromkeys(names, 0)
        for gid, size in zip(sub.index, sub["size"], strict=True):
            best = max(names, key=lambda s: (target[s] - current[s], ratios[s]))
            assign[int(gid)] = best
            current[best] += int(size)
    return df["group"].map(assign).to_numpy()
