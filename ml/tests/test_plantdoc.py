import pytest
from PIL import Image

from leafdx.data.classes import PLANTVILLAGE_CLASSES
from leafdx.data.plantdoc import build_manifest, load_mapping, normalize_name

# Folder names in pratikkayal/PlantDoc-Dataset@5467f60 (train/; test/ lacks spider mites),
# fetched from the GitHub contents API on 2026-11-12.
PLANTDOC_TRAIN_FOLDERS = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf spot",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato leaf",
    "Tomato mold leaf",
    "Tomato two spotted spider mites leaf",
    "grape leaf black rot",
    "grape leaf",
]


def test_mapping_covers_every_plantdoc_folder():
    m = load_mapping()
    assert set(m["key"]) == {normalize_name(f) for f in PLANTDOC_TRAIN_FOLDERS}
    assert set(m["plantvillage_class"]) <= set(PLANTVILLAGE_CLASSES)
    assert m["plantvillage_class"].is_unique  # no two PlantDoc classes merged into one


def test_normalize_name():
    assert normalize_name("Bell_pepper  leaf") == normalize_name("bell pepper leaf")


def test_build_manifest_rejects_unmapped_folder(tmp_path):
    root = tmp_path / "raw" / "plantdoc"
    for cls in ["Apple leaf", "Mystery leaf"]:
        d = root / "train" / cls
        d.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(d / "a.jpg")
    with pytest.raises(ValueError, match="Mystery leaf"):
        build_manifest(root, load_mapping(), tmp_path)


def test_build_manifest_rows(tmp_path):
    root = tmp_path / "raw" / "plantdoc"
    d = root / "test" / "Corn rust leaf"
    d.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(d / "a.jpg")
    (d / "notes.txt").write_text("x")
    df = build_manifest(root, load_mapping(), tmp_path)
    assert df.to_dict("records") == [
        {
            "path": "raw/plantdoc/test/Corn rust leaf/a.jpg",
            "plantdoc_split": "test",
            "plantdoc_class": "Corn rust leaf",
            "label": "Corn_(maize)___Common_rust_",
            "match": "approximate",
        }
    ]
