"""Recover provenance from New Plant Diseases Dataset (NPD) file names.

NPD was built by offline augmentation of PlantVillage and keeps the original file name with a
suffix per transform, e.g.::

    01a66316-...___FREC_Scab 3003.JPG                 original
    01a66316-...___FREC_Scab 3003_270deg.JPG          rotated copy
    01a66316-...___FREC_Scab 3003_new30degFlipLR.JPG  rotated 30 deg + mirrored

Stripping the suffix gives the *source image*. The upstream PlantVillage ``leaf-map.json`` maps
the photo identifier (the part after ``___``) to the *physical leaf*, since several photos were
often taken of one leaf. Both are used to group images before splitting.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

# Known families: 90deg, 180deg, 270deg, flipLR, flipTB, new30degFlipLR, new30degFlipTB,
# new90degFlipLR/TB, new200degFlipLR/TB, newGRR, newGGR and newPixel25.
#
# Split v1 (the one Phase 3 models were trained on) was built without the ``newPixel<n>``
# family, which this pattern originally missed: 2,399 such copies were treated as separate
# source photos. ``newpixel=False`` reproduces that behaviour exactly; see
# docs/dataset_card.md section 5 for the consequences.
_AUG_V1 = r"new\d+deg(?:Flip(?:LR|TB))?|new[A-Z]{2,5}|\d+deg|flip(?:LR|TB)"
_AUG_RE = re.compile(rf"_({_AUG_V1}|newPixel\d+)$")
_AUG_RE_V1 = re.compile(rf"_({_AUG_V1})$")

ORIGINAL = "original"


@dataclass(frozen=True)
class ParsedName:
    source_stem: str  # file stem with augmentation suffixes removed
    aug_type: str  # "original" or e.g. "270deg", "flipLR+90deg"


def parse_npd_filename(filename: str, newpixel: bool = True) -> ParsedName:
    """Strip augmentation suffixes. ``newpixel=False`` reproduces split v1's parsing."""
    pattern = _AUG_RE if newpixel else _AUG_RE_V1
    stem = PurePosixPath(filename).stem
    augs: list[str] = []
    while (m := pattern.search(stem)) is not None:
        augs.append(m.group(1))
        stem = stem[: m.start()]
    aug_type = "+".join(reversed(augs)) if augs else ORIGINAL
    return ParsedName(source_stem=stem.strip(), aug_type=aug_type)


def leaf_map_key(source_stem: str) -> str:
    """Lookup key used by upstream leaf-map.json (mirrors the official HF loader)."""
    ident = source_stem.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    return ident.strip().lower()


def resolve_leaf_id(
    source_stem: str, label: str, leaf_map: Mapping[str, Sequence[str]]
) -> str | None:
    """Return the upstream leaf id (``"<class>:::<n>"``) or None.

    Stricter than the official loader: a suggestion is accepted only if its class prefix equals
    the image's label, because identifiers such as ``RS_HL 6251`` recur across crops.
    """
    for suggestion in leaf_map.get(leaf_map_key(source_stem), ()):
        cls, _, _ = suggestion.partition(":::")
        if cls == label:
            return suggestion
    return None
