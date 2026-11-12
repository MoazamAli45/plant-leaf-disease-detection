import zipfile

from leafdx.data.download import extract_class_tree, locate_in_tree

ALL = lambda _c: True  # noqa: E731


def test_locate_in_nested_wrapper_folders():
    wrap = "New Plant Diseases Dataset(Augmented)"
    m = f"{wrap}/{wrap}/train/A___b/x.JPG"
    assert locate_in_tree(m, ["train", "valid"], ALL) == ("train", "A___b", "x.JPG")


def test_locate_rejects_other_layouts():
    assert locate_in_tree("test/test/AppleScab1.JPG", ["train", "valid"], ALL) is None
    assert locate_in_tree("train/x.JPG", ["train"], ALL) is None


def test_extract_dedupes_and_ignores_traversal(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Wrap/train/A___b/1.jpg", b"one")
        zf.writestr("wrap/train/A___b/1.jpg", b"one")  # identical duplicate tree
        zf.writestr("Wrap/valid/A___b/2.jpg", b"two")
        zf.writestr("Wrap/valid/Z___z/3.jpg", b"three")  # unknown class
        zf.writestr("Wrap/train/A___b/readme.txt", b"txt")
        zf.writestr("../train/A___b/evil.jpg", b"evil")  # traversal outside tree -> still safe
    out = tmp_path / "out"
    stats = extract_class_tree(archive, out, ["train", "valid"], known_classes={"A___b"})
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == ["train/A___b/1.jpg", "train/A___b/evil.jpg", "valid/A___b/2.jpg"]
    assert all(p.resolve().is_relative_to(out.resolve()) for p in out.rglob("*"))
    assert stats.duplicate_identical == 1
    assert stats.unknown_classes == {"Z___z": 1}
    assert stats.skipped_non_image == 1


def test_extract_renames_case_collisions(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train/A___b/CAR1.jpg", b"upper")
        zf.writestr("train/A___b/car1.jpg", b"lower")
    out = tmp_path / "out"
    stats = extract_class_tree(archive, out, ["train"])
    assert stats.written == 2
    assert stats.case_renamed == {"train/A___b/car1.jpg": "train/A___b/car1__case2.jpg"}
    assert (out / "train/A___b/CAR1.jpg").read_bytes() == b"upper"
    assert (out / "train/A___b/car1__case2.jpg").read_bytes() == b"lower"
