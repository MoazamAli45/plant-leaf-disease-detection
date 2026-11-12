"""Near-duplicate search with D4-invariant pHash distance.

``d4_distance(a, b) = min_k min(ham(a[0], b[k]), ham(a[k], b[0]))`` where ``a[k]`` is the hash
of the k-th dihedral transform of image a. Checking both directions makes the measure symmetric
(the DCT median threshold makes the two directions differ by a bit or two in rare cases).

Exhaustive search is O(n^2 * 16) popcounts. To keep it tractable:
- within one dataset we search **within each class** (NPD's augmented copies keep their label);
- across classes we only detect *exact* D4 matches via the canonical (minimum) hash;
- between datasets (PlantDoc vs NPD) the query set is small, so we search exhaustively.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

PAIR_COLUMNS = ["i", "j", "dist"]


def popcount(x: np.ndarray) -> np.ndarray:
    return np.bitwise_count(x)


def d4_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(m, n) uint8 D4 distances between hash sets a (m, 8) and b (n, 8)."""
    best = np.full((a.shape[0], b.shape[0]), 64, dtype=np.uint8)
    a0 = a[:, 0][:, None]
    b0 = b[:, 0][None, :]
    for k in range(a.shape[1]):
        np.minimum(best, popcount(a0 ^ b[:, k][None, :]), out=best)
        np.minimum(best, popcount(a[:, k][:, None] ^ b0), out=best)
    return best


def cross_pairs(a: np.ndarray, b: np.ndarray, max_dist: int, block: int = 512) -> pd.DataFrame:
    """All (i in a, j in b) with D4 distance <= max_dist."""
    out: list[np.ndarray] = []
    for s in range(0, len(a), block):
        d = d4_distance_matrix(a[s : s + block], b)
        ii, jj = np.nonzero(d <= max_dist)
        out.append(np.stack([ii + s, jj, d[ii, jj]], axis=1))
    arr = np.concatenate(out) if out else np.empty((0, 3))
    return pd.DataFrame(arr.astype(np.int64), columns=PAIR_COLUMNS)


def self_pairs(h: np.ndarray, max_dist: int, block: int = 512) -> pd.DataFrame:
    """All unordered pairs i < j within one hash set with D4 distance <= max_dist."""
    out: list[np.ndarray] = []
    for s in range(0, len(h), block):
        # only columns j > i are needed: compare the block against h[s:]
        d = d4_distance_matrix(h[s : s + block], h[s:])
        ii, jj = np.nonzero(d <= max_dist)
        jj_abs = jj + s
        ii_abs = ii + s
        keep = jj_abs > ii_abs
        out.append(np.stack([ii_abs[keep], jj_abs[keep], d[ii, jj][keep]], axis=1))
    arr = np.concatenate(out) if out else np.empty((0, 3))
    return pd.DataFrame(arr.astype(np.int64), columns=PAIR_COLUMNS)


def within_group_pairs(
    h: np.ndarray, group_labels: Sequence[str], max_dist: int, block: int = 512
) -> pd.DataFrame:
    """``self_pairs`` restricted to pairs sharing a label (indices refer to ``h``)."""
    labels = np.asarray(group_labels)
    frames = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        p = self_pairs(h[idx], max_dist, block)
        p["i"] = idx[p["i"].to_numpy()]
        p["j"] = idx[p["j"].to_numpy()]
        frames.append(p)
    if not frames:
        return pd.DataFrame(columns=PAIR_COLUMNS, dtype=np.int64)
    return pd.concat(frames, ignore_index=True)


def pairwise_d4(h: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """D4 distance for explicit index pairs (h[i[n]], h[j[n]])."""
    a, b = h[np.asarray(i)], h[np.asarray(j)]
    best = np.full(len(a), 64, dtype=np.uint8)
    for k in range(h.shape[1]):
        np.minimum(best, popcount(a[:, 0] ^ b[:, k]), out=best)
        np.minimum(best, popcount(a[:, k] ^ b[:, 0]), out=best)
    return best


def canonical_hash(h: np.ndarray) -> np.ndarray:
    """Minimum of the 8 transform hashes: equal for exact D4 copies."""
    return h.min(axis=1)


def exact_d4_pairs(h: np.ndarray) -> pd.DataFrame:
    """Pairs whose canonical hashes are identical (any label). Star-shaped per hash value."""
    canon = canonical_hash(h)
    order = np.argsort(canon, kind="stable")
    sorted_c = canon[order]
    same_as_prev = np.r_[False, sorted_c[1:] == sorted_c[:-1]]
    run_start = np.maximum.accumulate(np.where(~same_as_prev, np.arange(len(order)), 0))
    members = np.flatnonzero(same_as_prev)
    i = order[run_start[members]]
    j = order[members]
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    return pd.DataFrame({"i": lo, "j": hi, "dist": np.zeros(len(lo), dtype=np.int64)})
