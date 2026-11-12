import numpy as np
import pytest
from PIL import Image, ImageOps

from leafdx.data.hashing import index_image, phash_d4
from leafdx.data.neighbors import pairwise_d4


def textured(seed: int, size: int = 96) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (12, 12, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((size, size), Image.Resampling.BICUBIC)


def test_identity_hash_equals_imagehash():
    imagehash = pytest.importorskip("imagehash")
    img = textured(0)
    assert phash_d4(img)[0] == int(str(imagehash.phash(img)), 16)


@pytest.mark.parametrize(
    "transform",
    [
        lambda im: im.rotate(90, expand=True),
        lambda im: im.rotate(180),
        lambda im: im.rotate(270, expand=True),
        ImageOps.mirror,
        ImageOps.flip,
    ],
)
def test_d4_distance_small_for_dihedral_copies(transform):
    img = textured(1)
    h = np.array([phash_d4(img), phash_d4(transform(img))], dtype=np.uint64)
    assert pairwise_d4(h, np.array([0]), np.array([1]))[0] <= 2


def test_d4_distance_large_for_unrelated_images():
    h = np.array([phash_d4(textured(s)) for s in range(10)], dtype=np.uint64)
    i, j = np.triu_indices(10, k=1)
    assert pairwise_d4(h, i, j).min() > 8


def test_index_image_reports_corrupt_file(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    rec = index_image(bad)
    assert rec["error"]
    assert "sha256" in rec


def test_index_image_fields(tmp_path):
    p = tmp_path / "a.png"
    textured(2).save(p)
    rec = index_image(p)
    assert rec["error"] == ""
    assert rec["width"] == rec["height"] == 96
    assert len(rec["phash0"]) == 16
