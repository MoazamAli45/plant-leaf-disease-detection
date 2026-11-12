"""Pinned data sources. Every download is recorded in data/raw/PROVENANCE.json.

License notes (checked 2026-11-12; see docs/dataset_card.md for details and links):
- New Plant Diseases Dataset (Kaggle, vipoooool): offline-augmented re-release of PlantVillage.
  Kaggle license field: "Data files (c) Original Authors", i.e. upstream terms apply.
- PlantVillage (upstream, Hughes & Salathe 2015; Mohanty et al. 2016): the maintainers'
  Hugging Face card (mohanty/PlantVillage) declares CC BY-SA 3.0. Some third-party mirrors
  claim CC0 1.0; we follow the maintainers' stricter statement.
- PlantDoc (Singh et al. 2020): CC BY 4.0 (GitHub license API for pratikkayal/PlantDoc-Dataset).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KaggleSource:
    ref: str
    expected_version: int
    license: str
    homepage: str


@dataclass(frozen=True)
class UrlSource:
    url: str
    revision: str
    license: str
    homepage: str


NEW_PLANT_DISEASES = KaggleSource(
    ref="vipoooool/new-plant-diseases-dataset",
    expected_version=2,
    license="Data files (c) Original Authors (upstream PlantVillage: CC BY-SA 3.0)",
    homepage="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset",
)

PLANTVILLAGE_GIT_SHA = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
PLANTVILLAGE_LEAF_MAP = UrlSource(
    url=(
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/"
        f"{PLANTVILLAGE_GIT_SHA}/leaf_grouping/leaf-map.json"
    ),
    revision=PLANTVILLAGE_GIT_SHA,
    license="CC BY-SA 3.0 (per maintainers' Hugging Face card)",
    homepage="https://github.com/spMohanty/PlantVillage-Dataset",
)

PLANTDOC_GIT_SHA = "5467f6012d78d1c446145d5f582da6096f852ae8"
PLANTDOC = UrlSource(
    url=f"https://github.com/pratikkayal/PlantDoc-Dataset/archive/{PLANTDOC_GIT_SHA}.zip",
    revision=PLANTDOC_GIT_SHA,
    license="CC BY 4.0",
    homepage="https://github.com/pratikkayal/PlantDoc-Dataset",
)

KAGGLE_VIEW_API = "https://www.kaggle.com/api/v1/datasets/view/{ref}"
