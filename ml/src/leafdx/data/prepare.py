"""Build manifests and the leakage audit from the raw data.

    python -m leafdx.data.prepare --config configs/data.yaml

Steps
1. Scan NPD (paper split = Kaggle's train/valid folders) and parse provenance from file names.
2. Index every image (SHA-256, D4 pHash, statistics), cached in data/cache.
3. Find near-duplicate pairs (within class) and exact D4 copies (any class).
4. Audit the paper's 80/20 split for leakage.
5. Group images (source image | physical leaf | SHA-256 | pHash <= threshold) and make the
   grouped, stratified 70/15/15 clean split; verify it is leakage-free.
6. Build the PlantDoc field test set, excluding images that also occur in NPD.
7. Write manifests (ml/manifests) and reports (docs/figures/data_audit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from leafdx import __version__
from leafdx.data import audit, neighbors, plantdoc
from leafdx.data.classes import PLANTVILLAGE_CLASSES
from leafdx.data.download import IMAGE_SUFFIXES
from leafdx.data.filenames import parse_npd_filename, resolve_leaf_id
from leafdx.data.hashing import hashes_to_array, index_images
from leafdx.data.split import connected_groups, edges_from_keys, stratified_group_split

log = logging.getLogger(__name__)

PAPER_SPLIT_DIRS = {"train": "train", "valid": "val"}
MANIFEST_COLUMNS = ["path", "label", "group_id", "source_id", "leaf_id", "aug_type", "sha256"]


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    manifest_dir: Path = Path("manifests")
    report_dir: Path = Path("../docs/figures/data_audit")
    seed: int = 42
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    group_threshold: int = 4
    audit_thresholds: list[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12])
    plantdoc_overlap_threshold: int = 4
    drop_label_conflicts: bool = True
    drop_exact_duplicates: bool = True
    workers: int | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> DataConfig:
        raw = yaml.safe_load(path.read_text())
        return cls(
            data_dir=Path(raw["data_dir"]),
            manifest_dir=Path(raw["manifest_dir"]),
            report_dir=Path(raw["report_dir"]),
            seed=int(raw["seed"]),
            split_ratios={k: float(v) for k, v in raw["split_ratios"].items()},
            group_threshold=int(raw["phash"]["group_threshold"]),
            audit_thresholds=[int(t) for t in raw["phash"]["audit_thresholds"]],
            plantdoc_overlap_threshold=int(raw["plantdoc"]["overlap_threshold"]),
            drop_label_conflicts=bool(raw["drop_label_conflicts"]),
            drop_exact_duplicates=bool(raw["drop_exact_duplicates"]),
            workers=raw.get("workers"),
        )

    @property
    def max_dist(self) -> int:
        return max([*self.audit_thresholds, self.group_threshold, self.plantdoc_overlap_threshold])


# ----------------------------------------------------------------------------- scanning


def scan_npd(data_dir: Path, leaf_map: dict[str, list[str]]) -> pd.DataFrame:
    root = data_dir / "raw" / "npd"
    rows = []
    for folder, split in PAPER_SPLIT_DIRS.items():
        for cls in PLANTVILLAGE_CLASSES:
            d = root / folder / cls
            if not d.is_dir():
                raise FileNotFoundError(f"missing class folder {d}; run leafdx.data.download")
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                    parsed = parse_npd_filename(f.name)
                    rows.append(
                        {
                            "path": f.relative_to(data_dir).as_posix(),
                            "label": cls,
                            "paper_split": split,
                            "source_id": f"{cls}/{parsed.source_stem}",
                            "aug_type": parsed.aug_type,
                            "leaf_id": resolve_leaf_id(parsed.source_stem, cls, leaf_map),
                        }
                    )
    return pd.DataFrame(rows)


def attach_index(
    df: pd.DataFrame, data_dir: Path, cache: Path, workers: int | None
) -> pd.DataFrame:
    idx = index_images(data_dir, df["path"].tolist(), cache=cache, workers=workers)
    return df.merge(idx, on="path", how="left", validate="one_to_one")


def find_pairs(df: pd.DataFrame, h: np.ndarray, max_dist: int) -> pd.DataFrame:
    within = neighbors.within_group_pairs(h, df["label"].to_numpy(), max_dist)
    exact = neighbors.exact_d4_pairs(h)
    pairs = pd.concat([within, exact], ignore_index=True)
    return pairs.groupby(["i", "j"], as_index=False)["dist"].min()


# ----------------------------------------------------------------------------- reporting


def to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Minimal GitHub table writer (avoids a tabulate dependency)."""

    def fmt(v: Any) -> str:
        if isinstance(v, float | np.floating):
            return format(float(v), floatfmt)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def threshold_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "threshold_bits": int(t),
            "frac_near_duplicate_in_train": v["frac_near_duplicate_in_train"],
            "frac_not_explained_by_filename_or_leaf": v["frac_near_duplicate_not_explained"],
        }
        for t, v in summary["phash"].items()
    ]
    return pd.DataFrame(rows)


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


# ----------------------------------------------------------------------------- main pipeline


def run(cfg: DataConfig) -> dict[str, Any]:
    data_dir, rep, man = cfg.data_dir, cfg.report_dir, cfg.manifest_dir
    cache = data_dir / "cache"
    rep.mkdir(parents=True, exist_ok=True)
    leaf_map = json.loads((data_dir / "raw" / "plantvillage" / "leaf-map.json").read_text())

    # 1-2. scan + index
    df = scan_npd(data_dir, leaf_map)
    log.info("NPD: %d images", len(df))
    df = attach_index(df, data_dir, cache / "index_npd.csv", cfg.workers)
    unreadable = df[df["error"].astype(str) != ""]
    df = df[df["error"].astype(str) == ""].reset_index(drop=True)
    h = hashes_to_array(df)

    # 3. near-duplicate pairs
    pairs = find_pairs(df, h, cfg.max_dist)
    log.info("near-duplicate pairs (<= %d bits): %d", cfg.max_dist, len(pairs))

    # 4. audit of the paper split
    paper = audit.leakage_report(df, pairs, "paper_split", "val", "train", cfg.audit_thresholds)
    recall = audit.phash_recall_on_known_variants(df, h, cfg.audit_thresholds)

    # 5. groups + clean split
    close = pairs[pairs["dist"] <= cfg.group_threshold]
    edge_sets = {
        "source_image": edges_from_keys(df["source_id"]),
        "physical_leaf": edges_from_keys(df["leaf_id"]),
        "sha256": edges_from_keys(df["sha256"]),
        "phash": (close["i"].to_numpy(), close["j"].to_numpy()),
    }
    df["group_id"] = connected_groups(len(df), edge_sets.values())
    n_labels = df.groupby("group_id")["label"].nunique()
    conflict_groups = set(n_labels[n_labels > 1].index)
    df["label_conflict"] = df["group_id"].isin(conflict_groups)
    pool = pd.Series(True, index=df.index)
    if cfg.drop_label_conflicts:
        pool &= ~df["label_conflict"]
    if cfg.drop_exact_duplicates:
        pool &= ~df.sort_values("path").duplicated("sha256").reindex(df.index)
    df["clean_split"] = pd.Series(pd.NA, index=df.index, dtype=object)
    df.loc[pool, "clean_split"] = stratified_group_split(
        df.loc[pool, "label"], df.loc[pool, "group_id"], cfg.split_ratios, cfg.seed
    )
    clean = {
        s: audit.leakage_report(df, pairs, "clean_split", s, "train", cfg.audit_thresholds)
        for s in ("val", "test")
    }
    for s, r in clean.items():
        sm = r["summary"]
        within = sm["phash"][str(cfg.group_threshold)]["frac_near_duplicate_in_train"]
        bad = sm["frac_source_or_leaf_or_exact"] > 0 or within > 0
        if bad:
            raise AssertionError(f"clean {s} split leaks into train: {sm}")

    # 6. PlantDoc
    mapping = plantdoc.load_mapping()
    pdm = plantdoc.build_manifest(data_dir / "raw" / "plantdoc", mapping, data_dir)
    pdm = attach_index(pdm, data_dir, cache / "index_plantdoc.csv", cfg.workers)
    pd_unreadable = pdm[pdm["error"].astype(str) != ""]
    pdm = pdm[pdm["error"].astype(str) == ""].reset_index(drop=True)
    hp = hashes_to_array(pdm)
    cross = neighbors.cross_pairs(hp, h, cfg.max_dist)
    nn = cross.groupby("i")["dist"].min().reindex(range(len(pdm)))
    pdm["nearest_npd_dist"] = nn.astype("Int64")
    pdm["exclude_reason"] = ""
    pdm.loc[
        pdm["nearest_npd_dist"].le(cfg.plantdoc_overlap_threshold).fillna(False), "exclude_reason"
    ] = "near_duplicate_of_npd"
    dup = pdm.sort_values("path").duplicated("sha256").reindex(pdm.index)
    pdm.loc[dup & (pdm["exclude_reason"] == ""), "exclude_reason"] = (
        "exact_duplicate_within_plantdoc"
    )

    # 7. write manifests
    written: list[Path] = []
    for split in ("train", "val"):
        sub = df[df["paper_split"] == split].sort_values("path")
        written.append(write_csv(sub[["path", "label"]], man / "paper_split" / f"{split}.csv"))
    for split in cfg.split_ratios:
        sub = df[df["clean_split"] == split].sort_values("path")
        written.append(write_csv(sub[MANIFEST_COLUMNS], man / "clean_split" / f"{split}.csv"))
    pd_cols = [
        "path",
        "label",
        "plantdoc_class",
        "plantdoc_split",
        "match",
        "nearest_npd_dist",
        "exclude_reason",
        "sha256",
    ]
    written.append(write_csv(pdm.sort_values("path")[pd_cols], man / "plantdoc" / "all.csv"))
    ood = pdm[pdm["exclude_reason"] == ""].sort_values("path")
    written.append(write_csv(ood[pd_cols[:5]], man / "plantdoc" / "test.csv"))
    checksums = "".join(f"{sha256_text(p)}  {p.relative_to(man).as_posix()}\n" for p in written)
    (man / "SHA256SUMS").write_text(checksums)

    # caches for the EDA notebook
    df.to_csv(cache / "npd_full.csv", index=False)
    pairs.to_csv(cache / "npd_pairs.csv", index=False)
    pdm.to_csv(cache / "plantdoc_full.csv", index=False)

    # reports
    (rep / "paper_split_leakage.json").write_text(json.dumps(paper["summary"], indent=2))
    (rep / "paper_split_leakage_thresholds.md").write_text(
        to_markdown(threshold_table(paper["summary"]))
    )
    write_csv(paper["per_class"], rep / "paper_split_leakage_per_class.csv")
    write_csv(recall, rep / "phash_recall_by_aug_type.csv")
    (rep / "phash_recall_by_aug_type.md").write_text(to_markdown(recall, ".3f"))
    for s, r in clean.items():
        (rep / f"clean_split_{s}_leakage.json").write_text(json.dumps(r["summary"], indent=2))

    counts = (
        pd.concat(
            [
                df.groupby(["label", "paper_split"])
                .size()
                .unstack(fill_value=0)
                .add_prefix("paper_"),
                df.groupby(["label", "clean_split"])
                .size()
                .unstack(fill_value=0)
                .add_prefix("clean_"),
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    write_csv(counts, rep / "split_counts_per_class.csv")
    aug_counts = df.groupby(["aug_type", "paper_split"]).size().unstack(fill_value=0).reset_index()
    write_csv(aug_counts, rep / "aug_type_counts.csv")
    gsize = df.groupby("group_id").size()
    pd_summary = (
        pdm.assign(excluded=pdm["exclude_reason"] != "")
        .groupby(["plantdoc_class", "label", "match"])
        .agg(n=("path", "size"), n_excluded=("excluded", "sum"))
        .reset_index()
    )
    write_csv(pd_summary, rep / "plantdoc_classes.csv")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "leafdx_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "npd": {
            "n_images": int(len(df) + len(unreadable)),
            "n_unreadable": len(unreadable),
            "paper_split_counts": df["paper_split"].value_counts().to_dict(),
            "n_source_images": int(df["source_id"].nunique()),
            "n_with_leaf_id": int(df["leaf_id"].notna().sum()),
            "n_physical_leaves": int(df["leaf_id"].nunique()),
            "aug_type_counts": df["aug_type"].value_counts().to_dict(),
            "n_exact_duplicate_files": int(df.duplicated("sha256").sum()),
            "n_pairs_le_max_dist": len(pairs),
            "n_groups": int(gsize.size),
            "group_size": {
                "max": int(gsize.max()),
                "median": float(gsize.median()),
                "p99": float(gsize.quantile(0.99)),
            },
            "edges_per_link_type": {k: len(v[0]) for k, v in edge_sets.items()},
            "n_label_conflict_images": int(df["label_conflict"].sum()),
            "clean_split_counts": df["clean_split"].value_counts().to_dict(),
            "n_excluded_from_clean_split": int(df["clean_split"].isna().sum()),
        },
        "paper_split_audit": paper["summary"],
        "clean_split_audit": {s: r["summary"] for s, r in clean.items()},
        "plantdoc": {
            "n_images": int(len(pdm) + len(pd_unreadable)),
            "n_unreadable": len(pd_unreadable),
            "excluded": pdm["exclude_reason"].replace("", "kept").value_counts().to_dict(),
            "n_test": len(ood),
            "n_classes_test": int(ood["label"].nunique()),
        },
        "manifest_sha256": {p.relative_to(man).as_posix(): sha256_text(p) for p in written},
    }
    (rep / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("done: %s", json.dumps(summary["npd"]["clean_split_counts"]))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--seed", type=int, help="override the split seed")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = DataConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    run(cfg)


if __name__ == "__main__":
    main()
