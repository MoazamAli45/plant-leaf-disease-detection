"""EDA and audit figures, generated from the caches written by ``leafdx.data.prepare``.

    python -m leafdx.eda --config configs/data.yaml

Writes PNGs and CSV tables to docs/figures/eda/. The EDA notebook calls the same functions.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from leafdx.data import audit
from leafdx.data.classes import PLANTVILLAGE_CLASSES, display_name
from leafdx.data.filenames import ORIGINAL
from leafdx.data.prepare import DataConfig

log = logging.getLogger(__name__)

# Reference palette (dataviz skill), light mode; validated: blue/orange pass all checks.
SERIES = ["#2a78d6", "#eb6834"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 9,
        "legend.frameon": False,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def load(cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = cfg.data_dir / "cache"
    npd = pd.read_csv(cache / "npd_full.csv", keep_default_na=False, na_values=[""])
    pairs = pd.read_csv(cache / "npd_pairs.csv")
    pdoc = pd.read_csv(cache / "plantdoc_full.csv", keep_default_na=False, na_values=[""])
    return npd, pairs, pdoc


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_class_counts(npd: pd.DataFrame, out: Path) -> Path:
    """Images per class in the paper's train/val split (stacked horizontal bars)."""
    c = npd.groupby(["label", "paper_split"]).size().unstack(fill_value=0)
    c = c.reindex(PLANTVILLAGE_CLASSES).fillna(0)
    c = c.loc[c.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7.5, 9))
    y = np.arange(len(c))
    left = np.zeros(len(c))
    for colour, split in zip(SERIES, ["train", "val"], strict=True):
        vals = c.get(split, pd.Series(0, index=c.index)).to_numpy()
        ax.barh(
            y,
            vals,
            left=left,
            height=0.7,
            color=colour,
            edgecolor=SURFACE,
            linewidth=1,
            label=f"paper {split}",
        )
        left += vals
    ax.set_yticks(y, [display_name(n) for n in c.index], fontsize=7.5)
    ax.set_xlabel("images")
    ax.grid(axis="y", visible=False)
    ax.set_title("New Plant Diseases Dataset: images per class")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, borderaxespad=1.6)
    return _save(fig, out, "class_counts.png")


def fig_aug_types(npd: pd.DataFrame, out: Path) -> Path:
    c = npd["aug_type"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(c.index, c.to_numpy(), height=0.6, color=SERIES[0])
    for i, v in enumerate(c.to_numpy()):
        ax.text(v, i, f" {v:,}", va="center", color=INK_2, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("images")
    ax.set_title("Offline augmentation types (from file names)")
    return _save(fig, out, "aug_type_counts.png")


def _grid(paths: list[Path], titles: list[str], ncols: int, title: str) -> plt.Figure:
    nrows = int(np.ceil(len(paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.85))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, p, t in zip(np.atleast_1d(axes).ravel(), paths, titles, strict=False):
        with Image.open(p) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(textwrap.fill(t, 22), fontsize=6, color=INK, loc="center", fontweight="normal")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    return fig


def fig_sample_grid_npd(npd: pd.DataFrame, data_dir: Path, out: Path, seed: int) -> Path:
    orig = npd[npd["aug_type"] == ORIGINAL]
    pick = orig.groupby("label").sample(1, random_state=seed).set_index("label")
    labels = [c for c in PLANTVILLAGE_CLASSES if c in pick.index]
    fig = _grid(
        [data_dir / pick.loc[c, "path"] for c in labels],
        [display_name(c) for c in labels],
        8,
        "New Plant Diseases Dataset: one original image per class",
    )
    return _save(fig, out, "samples_npd.png")


def fig_sample_grid_plantdoc(pdoc: pd.DataFrame, data_dir: Path, out: Path, seed: int) -> Path:
    pick = pdoc.groupby("plantdoc_class").sample(1, random_state=seed)
    pick = pick.sort_values("plantdoc_class", key=lambda c: c.str.casefold())
    fig = _grid(
        [data_dir / p for p in pick["path"]],
        list(pick["plantdoc_class"]),
        7,
        "PlantDoc (field images): one image per class",
    )
    return _save(fig, out, "samples_plantdoc.png")


def fig_augmented_variants(
    npd: pd.DataFrame, data_dir: Path, out: Path, seed: int, n_sources: int = 4
) -> Path | None:
    """Rows = source images; columns = the original and its offline-augmented copies."""
    by_src = npd.groupby("source_id")
    has_orig = by_src["aug_type"].apply(lambda s: (s == ORIGINAL).any() and len(s) >= 2)
    if not has_orig.any():
        return None
    rng = np.random.default_rng(seed)
    srcs = rng.choice(
        has_orig[has_orig].index.to_numpy(), size=min(n_sources, has_orig.sum()), replace=False
    )
    rows = [
        npd[npd["source_id"] == s].sort_values("aug_type", key=lambda a: a != ORIGINAL)
        for s in srcs
    ]
    ncols = max(len(r) for r in rows)
    fig, axes = plt.subplots(
        len(rows), ncols, figsize=(ncols * 1.7, len(rows) * 1.9), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")
    for r, sub in enumerate(rows):
        for c, (_, row) in enumerate(sub.iterrows()):
            with Image.open(data_dir / row["path"]) as im:
                axes[r, c].imshow(im.convert("RGB"))
            axes[r, c].set_title(
                f"{row['aug_type']}\n({row['paper_split']})",
                fontsize=7,
                loc="center",
                fontweight="normal",
            )
    fig.suptitle(
        "Augmented copies of one source image, and the paper split each landed in",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    return _save(fig, out, "augmented_variants.png")


def leakage_curve(
    npd: pd.DataFrame, pairs: pd.DataFrame, split_col: str, eval_split: str, thresholds: list[int]
) -> pd.Series:
    hits = audit.phash_overlap(npd, pairs, split_col, eval_split, "train", thresholds)
    return pd.Series({t: float(h.mean()) if len(h) else 0.0 for t, h in hits.items()})


def fig_leakage_curve(npd: pd.DataFrame, pairs: pd.DataFrame, cfg: DataConfig, out: Path) -> Path:
    t = sorted(cfg.audit_thresholds)
    paper = leakage_curve(npd, pairs, "paper_split", "val", t)
    clean = leakage_curve(npd, pairs, "clean_split", "test", t)
    pd.DataFrame(
        {"threshold_bits": t, "paper_val": paper.to_numpy(), "clean_test": clean.to_numpy()}
    ).to_csv(out / "leakage_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for s, colour, marker, name in [
        (paper, SERIES[0], "o", "paper 80/20 val"),
        (clean, SERIES[1], "s", "clean 70/15/15 test"),
    ]:
        ax.plot(s.index, s.to_numpy() * 100, color=colour, lw=2, marker=marker, ms=6, label=name)
        ax.annotate(
            f"{name}: {s.iloc[-1] * 100:.1f}%",
            (s.index[-1], s.iloc[-1] * 100),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=8,
        )
    ax.axvline(cfg.group_threshold, color=INK_2, lw=0.8, ls=":")
    ax.text(
        cfg.group_threshold,
        ax.get_ylim()[1],
        " grouping threshold",
        color=INK_2,
        fontsize=7,
        va="top",
    )
    ax.set_xlabel("D4 pHash distance threshold (bits of 64)")
    ax.set_ylabel("eval images with a train near-duplicate (%)")
    ax.set_title("Perceptual near-duplicates of training images")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncols=2)
    return _save(fig, out, "leakage_curve.png")


def fig_hash_only_pairs(
    npd: pd.DataFrame, pairs: pd.DataFrame, data_dir: Path, out: Path, seed: int, per_dist: int = 2
) -> Path | None:
    """Near-duplicate pairs NOT explained by file name or leaf id, by distance: used to pick
    and sanity-check the grouping threshold by eye."""
    src = npd["source_id"].to_numpy()
    leaf = npd["leaf_id"].fillna("").to_numpy()
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    unexplained = pairs[(src[i] != src[j]) & ~((leaf[i] != "") & (leaf[i] == leaf[j]))]
    if unexplained.empty:
        return None
    pick = unexplained.groupby("dist", group_keys=False).apply(
        lambda g: g.sample(min(per_dist, len(g)), random_state=seed)
    )
    pick = pick.sort_values("dist").head(16)
    fig, axes = plt.subplots(len(pick), 2, figsize=(3.6, 1.8 * len(pick)), squeeze=False)
    for r, (_, p) in enumerate(pick.iterrows()):
        for c, idx in enumerate((p["i"], p["j"])):
            row = npd.iloc[int(idx)]
            with Image.open(data_dir / row["path"]) as im:
                axes[r, c].imshow(im.convert("RGB"))
            axes[r, c].axis("off")
            axes[r, c].set_title(
                f"d={p['dist']}  {display_name(row['label'])[:28]}",
                fontsize=6,
                loc="center",
                fontweight="normal",
            )
    fig.suptitle(
        "pHash pairs not explained by file name or leaf id",
        x=0.01,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    return _save(fig, out, "hash_only_pairs.png")


def fig_background_brightness(npd: pd.DataFrame, pdoc: pd.DataFrame, out: Path) -> Path:
    """Border (background proxy) brightness: lab images vs field images."""
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.linspace(0, 255, 52)
    for vals, colour, name in [
        (npd["border_mean"], SERIES[0], "NPD (lab)"),
        (pdoc["border_mean"], SERIES[1], "PlantDoc (field)"),
    ]:
        w = np.ones(len(vals)) / max(len(vals), 1) * 100
        ax.hist(vals, bins=bins, weights=w, histtype="step", lw=2, color=colour, label=name)
    ax.set_xlabel("mean grayscale of the 4-px image border (64x64 thumbnail)")
    ax.set_ylabel("% of images")
    ax.set_title("Background brightness: lab vs field images")
    ax.legend()
    return _save(fig, out, "background_brightness.png")


def table_image_stats(npd: pd.DataFrame, pdoc: pd.DataFrame, out: Path) -> Path:
    rows = []
    for name, df in [("npd", npd), ("plantdoc", pdoc)]:
        size = df["width"].astype(int).astype(str) + "x" + df["height"].astype(int).astype(str)
        top = size.value_counts()
        rows.append(
            {
                "dataset": name,
                "n_images": len(df),
                "n_distinct_sizes": int(top.size),
                "most_common_size": top.index[0],
                "frac_most_common_size": top.iloc[0] / len(df),
                "median_width": float(df["width"].median()),
                "median_height": float(df["height"].median()),
                "mean_r": df["mean_r"].mean(),
                "mean_g": df["mean_g"].mean(),
                "mean_b": df["mean_b"].mean(),
                "mean_border_brightness": df["border_mean"].mean(),
                "mean_file_kb": df["bytes"].mean() / 1024,
            }
        )
    p = out / "image_stats.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def run(cfg: DataConfig, out: Path | None = None) -> list[Path]:
    out = out or cfg.report_dir.parent / "eda"
    out.mkdir(parents=True, exist_ok=True)
    npd, pairs, pdoc = load(cfg)
    d, s = cfg.data_dir, cfg.seed
    made = [
        fig_class_counts(npd, out),
        fig_aug_types(npd, out),
        fig_sample_grid_npd(npd, d, out, s),
        fig_sample_grid_plantdoc(pdoc, d, out, s),
        fig_augmented_variants(npd, d, out, s),
        fig_leakage_curve(npd, pairs, cfg, out),
        fig_hash_only_pairs(npd, pairs, d, out, s),
        fig_background_brightness(npd, pdoc, out),
        table_image_stats(npd, pdoc, out),
    ]
    return [p for p in made if p is not None]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for path in run(DataConfig.from_yaml(args.config)):
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
