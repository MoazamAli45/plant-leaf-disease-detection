"""FieldPlant leaf crops for Phase 7 (protocol: docs/field2.md).

    python -m leafdx.data.extra_field fieldplant

FieldPlant (Moupojou et al., IEEE Access 2023; CC BY 4.0) has 5,156 photos taken in
plantations in Cameroon, with 8,580 leaves annotated by bounding box under the supervision of
plant pathologists. Its class names follow PlantDoc's, so the same mapping rules are used
(``resources/fieldplant_to_plantvillage.csv``); classes without a PlantVillage counterpart
(cassava, most corn disorders) and the two ambiguous tomato classes are left out.

Each mapped leaf box is cropped with a 5% margin and saved with its longer side at most 640 px.
Crops from one photo, and from near-duplicate photos (D4 pHash distance <= 6 bits, as for
PlantDoc), form one group; groups are split 50/15/35 into train/val/test, stratified by class.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from leafdx.data.hashing import hashes_to_array, index_images
from leafdx.data.neighbors import self_pairs
from leafdx.data.split import connected_groups, stratified_group_split

ARCHIVE = Path("data/downloads/fieldplant/fieldplant.zip")
MAPPING = Path("resources/fieldplant_to_plantvillage.csv")
OUT = Path("manifests/fieldplant")
RATIOS = {"train": 0.50, "val": 0.15, "test": 0.35}
MARGIN = 0.05
MAX_SIDE = 640
MAX_DIST = 6
SEED = 42


def _crop(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = box
    mx, my = MARGIN * (x1 - x0), MARGIN * (y1 - y0)
    c = img.crop(
        (
            max(0, int(x0 - mx)),
            max(0, int(y0 - my)),
            min(img.width, int(x1 + mx)),
            min(img.height, int(y1 + my)),
        )
    )
    c.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return c


def prepare(data_root: Path) -> Path:
    import hashlib

    mapping = pd.read_csv(MAPPING)
    to_pv = dict(zip(mapping["fieldplant_class"], mapping["plantvillage_class"], strict=True))
    raw = data_root / "raw" / "fieldplant"
    src_dir = raw / "photos"
    with zipfile.ZipFile(ARCHIVE) as z:
        ann = pd.read_csv(z.open("train/_annotations.csv")).rename(columns={"class": "cls"})
        ann = ann[ann["cls"].isin(to_pv)].reset_index(drop=True)
        photos = sorted(ann["filename"].unique())
        src_dir.mkdir(parents=True, exist_ok=True)
        for f in photos:
            dest = src_dir / f
            if not dest.exists():
                dest.write_bytes(z.read(f"train/{f}"))

    # group near-duplicate photos (Roboflow exports can hold the same photo more than once)
    idx = index_images(src_dir, photos, cache=Path("data/cache/index_fieldplant.csv"))
    idx = idx.set_index("path").loc[photos].reset_index()
    pairs = self_pairs(hashes_to_array(idx), MAX_DIST)
    photo_group = connected_groups(len(photos), [(pairs["i"].to_numpy(), pairs["j"].to_numpy())])
    group_of = dict(zip(photos, photo_group, strict=True))

    rows = []
    for f, boxes in ann.groupby("filename", sort=True):
        img = ImageOps.exif_transpose(Image.open(src_dir / f)).convert("RGB")
        for i, b in enumerate(boxes.itertuples(index=False)):
            label = to_pv[b.cls]
            rel = Path("raw/fieldplant/crops") / label.replace(" ", "_") / f"{Path(f).stem}_{i}.jpg"
            dest = data_root / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                _crop(img, (b.xmin, b.ymin, b.xmax, b.ymax)).save(dest, quality=95)
            rows.append(
                {
                    "path": str(rel),
                    "label": label,
                    "fieldplant_class": b.cls,
                    "source_image": f,
                    "group": int(group_of[f]),
                    "box_w": int(b.xmax - b.xmin),
                    "box_h": int(b.ymax - b.ymin),
                }
            )
    df = pd.DataFrame(rows)
    df["split"] = stratified_group_split(df["label"], df["group"], RATIOS, SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    for name in RATIOS:
        df[df["split"] == name].drop(columns="split").to_csv(OUT / f"{name}.csv", index=False)
    per_class = df.groupby(["label", "split"]).size().unstack(fill_value=0)[list(RATIOS)]
    report = Path("../docs/figures/phase7")
    report.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(report / "fieldplant_split_per_class.csv")
    summary = {
        "archive": str(ARCHIVE),
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "n_photos_used": len(photos),
        "n_crops": len(df),
        "n_photo_pairs_linked": len(pairs),
        "n_groups": int(df["group"].nunique()),
        "groups_crossing_splits": int((df.groupby("group")["split"].nunique() > 1).sum()),
        "counts": df["split"].value_counts().reindex(list(RATIOS)).to_dict(),
        "margin": MARGIN,
        "max_side": MAX_SIDE,
        "max_dist": MAX_DIST,
        "seed": SEED,
    }
    (report / "fieldplant_split.json").write_text(json.dumps(summary, indent=1) + "\n")
    prov = data_root / "raw" / "PROVENANCE.json"
    p = json.loads(prov.read_text())
    p["fieldplant"] = {
        "source": {
            "url": "https://www.kaggle.com/datasets/bloox2/fieldplant",
            "original": "https://universe.roboflow.com/plant-disease-detection/fieldplant (v11)",
            "license": "CC BY 4.0 (README.dataset.txt in the archive)",
            "citation": "Moupojou et al., IEEE Access 11, 2023, doi:10.1109/ACCESS.2023.3263042",
        },
        "archive_sha256": summary["archive_sha256"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    prov.write_text(json.dumps(p, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(per_class.to_string())
    return OUT
