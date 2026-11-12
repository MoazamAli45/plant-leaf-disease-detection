import numpy as np
import pandas as pd
import pytest

from leafdx.data.split import connected_groups, edges_from_keys, stratified_group_split

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def synthetic(n_groups: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 6, n_groups)
    groups = np.repeat(np.arange(n_groups), sizes)
    group_label = rng.choice(["a", "b", "c", "d"], n_groups, p=[0.5, 0.3, 0.15, 0.05])
    return group_label[groups], groups


def test_groups_never_straddle_splits():
    labels, groups = synthetic()
    s = stratified_group_split(labels, groups, RATIOS, seed=1)
    assert pd.DataFrame({"g": groups, "s": s}).groupby("g")["s"].nunique().max() == 1


def test_per_class_proportions_close_to_target():
    labels, groups = synthetic()
    s = stratified_group_split(labels, groups, RATIOS, seed=1)
    frac = pd.crosstab(labels, s, normalize="index")
    for split, r in RATIOS.items():
        assert np.allclose(frac[split], r, atol=0.03), frac


def test_deterministic_and_seed_dependent():
    labels, groups = synthetic()
    a = stratified_group_split(labels, groups, RATIOS, seed=7)
    b = stratified_group_split(labels, groups, RATIOS, seed=7)
    c = stratified_group_split(labels, groups, RATIOS, seed=8)
    assert (a == b).all()
    assert (a != c).any()


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        stratified_group_split(["a"], [0], {"train": 0.5, "val": 0.2}, seed=0)


def test_edges_from_keys_and_components():
    keys = ["x", None, "y", "x", "", "y", "z"]
    i, j = edges_from_keys(keys)
    assert sorted(zip(i.tolist(), j.tolist(), strict=True)) == [(0, 3), (2, 5)]
    comp = connected_groups(7, [(i, j), (np.array([3]), np.array([6]))])
    assert comp[0] == comp[3] == comp[6]
    assert comp[2] == comp[5] != comp[0]
    assert len(set(comp.tolist())) == 4  # {0,3,6}, {2,5}, {1}, {4}
