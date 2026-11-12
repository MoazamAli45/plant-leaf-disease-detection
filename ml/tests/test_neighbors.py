import numpy as np

from leafdx.data.neighbors import (
    cross_pairs,
    d4_distance_matrix,
    exact_d4_pairs,
    self_pairs,
    within_group_pairs,
)


def random_hashes(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**63, size=(n, 8), dtype=np.uint64)


def naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((len(a), len(b)), dtype=int)
    for x in range(len(a)):
        for y in range(len(b)):
            out[x, y] = min(
                min(bin(int(a[x, 0]) ^ int(b[y, k])).count("1") for k in range(8)),
                min(bin(int(a[x, k]) ^ int(b[y, 0])).count("1") for k in range(8)),
            )
    return out


def test_distance_matrix_matches_naive():
    a, b = random_hashes(7, 1), random_hashes(5, 2)
    np.testing.assert_array_equal(d4_distance_matrix(a, b), naive(a, b))


def test_self_pairs_finds_planted_near_duplicates_across_blocks():
    h = random_hashes(50)
    h[37] = h[3] ^ np.uint64(0b101)  # 2 bits apart in every transform
    h[49] = h[10]
    p = self_pairs(h, max_dist=4, block=8)
    found = {(int(i), int(j)): int(d) for i, j, d in p.itertuples(index=False)}
    assert found == {(3, 37): 2, (10, 49): 0}


def test_within_group_pairs_ignores_other_labels():
    h = random_hashes(6)
    h[5] = h[0]
    h[4] = h[1]
    labels = ["a", "b", "a", "b", "a", "a"]  # (0,5) same label, (1,4) different labels
    p = within_group_pairs(h, labels, max_dist=0)
    assert sorted(map(tuple, p[["i", "j"]].to_numpy().tolist())) == [(0, 5)]


def test_exact_d4_pairs_any_label():
    h = random_hashes(6)
    h[4] = h[1][::-1]  # same set of transform hashes, different order -> same canonical hash
    p = exact_d4_pairs(h)
    assert p[["i", "j"]].to_numpy().tolist() == [[1, 4]]


def test_cross_pairs():
    a, b = random_hashes(3, 5), random_hashes(20, 6)
    a[2] = b[17]
    p = cross_pairs(a, b, max_dist=3, block=2)
    assert p[["i", "j", "dist"]].to_numpy().tolist() == [[2, 17, 0]]
