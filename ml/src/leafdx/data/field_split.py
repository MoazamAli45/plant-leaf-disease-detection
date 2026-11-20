"""Grouped split of the PlantDoc field images for Phase 6 (field adaptation).

    python -m leafdx.data.field_split            # -> manifests/field_split/, docs/figures/phase6/

Until Phase 5 every PlantDoc image was test data. Methods that learn from field images need a
separate, untouched field test set, so the 2,557 cleaned images are split into
``train`` (labelled examples a method may use), ``val`` (model and hyper-parameter selection)
and ``test`` (reported once, at the end).

PlantDoc was collected from the web and contains copies of the same picture (resized,
re-compressed, cropped, sometimes filed under two classes), and its own train/test split
places some copies on both sides. Images are therefore grouped before splitting: two images
are linked when their D4-invariant pHash distance is at most ``--max-dist`` bits (6 by default;
chosen by inspecting pairs, see docs/field.md), and each connected group goes to one split.
The split is stratified by class and deterministic for a given seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from leafdx.data.hashing import hashes_to_array
from leafdx.data.neighbors import self_pairs
from leafdx.data.split import connected_groups, stratified_group_split

RATIOS = {"train": 0.50, "val": 0.15, "test": 0.35}


def build(manifest: Path, index: Path, max_dist: int, seed: int) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(manifest)
    idx = pd.read_csv(index)
    hashed = df.merge(idx, on="path", how="left", validate="one_to_one")
    if hashed["phash0"].isna().any():
        raise ValueError("some PlantDoc images have no hash in the index")
    pairs = self_pairs(hashes_to_array(hashed), max_dist)
    groups = connected_groups(len(df), [(pairs["i"].to_numpy(), pairs["j"].to_numpy())])
    df["group"] = groups
    df["field_split"] = stratified_group_split(df["label"], groups, RATIOS, seed)

    labels_per_group = df.groupby("group")["label"].nunique()
    sizes = df.groupby("group").size()
    leaks_in_original = int(
        (df.groupby("group")["plantdoc_split"].nunique() > 1).sum()
    )  # groups that PlantDoc's own split cuts in two
    summary = {
        "source": str(manifest),
        "max_dist": max_dist,
        "seed": seed,
        "ratios": RATIOS,
        "n_images": len(df),
        "n_classes": int(df["label"].nunique()),
        "n_pairs_linked": len(pairs),
        "n_groups": int(groups.max() + 1),
        "n_groups_multi": int((sizes > 1).sum()),
        "largest_group": int(sizes.max()),
        "n_groups_with_conflicting_labels": int((labels_per_group > 1).sum()),
        "n_groups_split_by_plantdoc_original_split": leaks_in_original,
        "counts": df["field_split"].value_counts().reindex(list(RATIOS)).to_dict(),
        "min_train_images_per_class": int(
            df[df["field_split"] == "train"].groupby("label").size().min()
        ),
    }
    return df, summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("manifests/plantdoc/test.csv"))
    p.add_argument("--index", type=Path, default=Path("data/cache/index_plantdoc.csv"))
    p.add_argument("--out", type=Path, default=Path("manifests/field_split"))
    p.add_argument("--report", type=Path, default=Path("../docs/figures/phase6"))
    p.add_argument("--max-dist", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    df, summary = build(a.manifest, a.index, a.max_dist, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    a.report.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "plantdoc_class", "plantdoc_split", "match", "group"]
    for name in RATIOS:
        df.loc[df["field_split"] == name, cols].to_csv(a.out / f"{name}.csv", index=False)
    per_class = df.groupby(["label", "field_split"]).size().unstack(fill_value=0)[list(RATIOS)]
    per_class.to_csv(a.report / "field_split_per_class.csv")
    (a.report / "field_split.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(per_class.describe().loc[["min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
