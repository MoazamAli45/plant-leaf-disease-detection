"""PlantDoc (field images) as an out-of-distribution test set mapped to PlantVillage labels."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from leafdx.data.classes import PLANTVILLAGE_CLASSES
from leafdx.data.download import IMAGE_SUFFIXES

DEFAULT_MAPPING = Path(__file__).resolve().parents[3] / "resources" / "plantdoc_to_plantvillage.csv"
MATCH_TYPES = {"exact", "healthy_assumed", "approximate"}


def normalize_name(name: str) -> str:
    """Case-, underscore- and whitespace-insensitive key for folder names."""
    return re.sub(r"[\s_]+", " ", name).strip().casefold()


def load_mapping(path: Path = DEFAULT_MAPPING) -> pd.DataFrame:
    m = pd.read_csv(path)
    bad_targets = set(m["plantvillage_class"]) - set(PLANTVILLAGE_CLASSES)
    if bad_targets:
        raise ValueError(f"mapping targets unknown PlantVillage classes: {sorted(bad_targets)}")
    bad_match = set(m["match"]) - MATCH_TYPES
    if bad_match:
        raise ValueError(f"unknown match types: {sorted(bad_match)}")
    keys = m["plantdoc_class"].map(normalize_name)
    if keys.duplicated().any():
        raise ValueError(
            f"duplicate PlantDoc classes: {m.loc[keys.duplicated(), 'plantdoc_class']}"
        )
    return m.assign(key=keys)


def build_manifest(root: Path, mapping: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """One row per PlantDoc image under ``root/{train,test}/<class>/``.

    Raises if a class folder is not in the mapping, so a new PlantDoc release cannot silently
    drop or mislabel classes. Paths are relative to ``data_root``.
    """
    by_key = mapping.set_index("key")
    rows = []
    unmapped: set[str] = set()
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            key = normalize_name(cls_dir.name)
            if key not in by_key.index:
                unmapped.add(cls_dir.name)
                continue
            m = by_key.loc[key]
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                    rows.append(
                        {
                            "path": f.relative_to(data_root).as_posix(),
                            "plantdoc_split": split_dir.name,
                            "plantdoc_class": cls_dir.name,
                            "label": m["plantvillage_class"],
                            "match": m["match"],
                        }
                    )
    if unmapped:
        raise ValueError(f"PlantDoc folders missing from mapping: {sorted(unmapped)}")
    return pd.DataFrame(rows)
