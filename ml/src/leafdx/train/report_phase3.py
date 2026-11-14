"""Phase 3 tables and figures, generated from ``results/p3_*`` (mean and std over seeds).

python -m leafdx.train.report_phase3 --results results --out ../docs/figures/phase3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leafdx.data.classes import PLANTVILLAGE_CLASSES, display_name, is_healthy, split_class

MAIN = {
    "p3_cnn_softmax": "Paper CNN (softmax)",
    "p3_mobilenet_v1": "MobileNetV1",
    "p3_mobilenet_v2": "MobileNetV2",
    "p3_mobilenet_v3_small": "MobileNetV3-Small",
    "p3_efficientnet_b0": "EfficientNetB0",
}
ABLATIONS = {
    "p3_mobilenet_v1": "MobileNetV1 (main: fine-tuned, augmentation, alpha 1.0)",
    "p3_abl_mnv1_noaug": "no augmentation",
    "p3_abl_mnv1_frozen": "frozen backbone (as in the paper)",
    "p3_abl_mnv1_alpha075": "width multiplier 0.75",
    "p3_abl_mnv1_alpha050": "width multiplier 0.5",
}
SERIES = ["#2a78d6", "#eb6834"]  # dataviz reference palette (validated pair)
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
THRESHOLDS = [0.5, 0.7, 0.9]
# Images of clean split v1's test set with a source-photo/leaf link to train under the
# corrected parser (see leafdx.data.recheck); excluded for the leak-free accuracy.
LINKS = Path("manifests/clean_split/test_links_corrected.csv")
_LINKED: dict[str, bool] | None = None


def linked_test_paths() -> dict[str, bool]:
    global _LINKED
    if _LINKED is None:
        lk = pd.read_csv(LINKS) if LINKS.exists() else pd.DataFrame(columns=["path", "linked_any"])
        _LINKED = dict(zip(lk["path"], lk["linked_any"].astype(bool), strict=True))
    return _LINKED


def leakfree_test_accuracy(run: dict[str, Any]) -> float:
    d = np.load(run["_dir"] / "probs_test.npz")
    linked = linked_test_paths()
    keep = np.array([not linked.get(str(p), False) for p in d["paths"]])
    return float((d["probs"].argmax(1) == d["labels"])[keep].mean())


# ----------------------------------------------------------------------------- loading


def load(results: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(results.glob("p3_*/seed*/metrics.json")):
        m = json.loads(f.read_text())
        m["_dir"] = f.parent
        runs.setdefault(m["name"], []).append(m)
    return runs


def _probs(run: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(run["_dir"] / f"probs_{split}.npz")
    return d["probs"].astype(np.float32), d["labels"]


def _ms(values: list[float], pct: bool = True) -> str:
    v = np.asarray(values, dtype=float) * (100 if pct else 1)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.2f} ± {sd:.2f}"


def _md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows]) + "\n"


# ----------------------------------------------------------------------------- tables


def summary_metrics(m: dict[str, Any]) -> dict[str, float]:
    e = m["eval"]
    t, p = e["test"], e["plantdoc"]
    return {
        "test_acc": t["accuracy"],
        "test_acc_leakfree": leakfree_test_accuracy(m),
        "test_orig_acc": t["subsets"]["original"]["accuracy"],
        "test_macro_f1": t["macro_f1"],
        "test_ece": t["ece"],
        "pd_acc": p["accuracy"],
        "pd_restricted_acc": p["restricted_accuracy"],
        "pd_macro_f1": p["macro_f1"],
        "pd_exact_acc": p["subsets"]["exact"]["accuracy"],
        "pd_ece": p["ece"],
        "gap": leakfree_test_accuracy(m) - p["accuracy"],
    }


COLUMNS = {
    "test_acc": "test acc. (%)",
    "test_acc_leakfree": "test acc., leak-free subset (%)",
    "test_orig_acc": "test acc., originals only (%)",
    "test_macro_f1": "test macro-F1 (%)",
    "test_ece": "test ECE (%)",
    "pd_acc": "PlantDoc acc. (%)",
    "pd_restricted_acc": "PlantDoc acc., 28 classes (%)",
    "pd_macro_f1": "PlantDoc macro-F1 (%)",
    "pd_exact_acc": "PlantDoc acc., exact-match classes (%)",
    "pd_ece": "PlantDoc ECE (%)",
    "gap": "gap leak-free test - PlantDoc (pp)",
}


def results_table(runs: dict[str, list[dict[str, Any]]], names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for key, label in names.items():
        if key not in runs:
            continue
        ms = [summary_metrics(m) for m in runs[key]]
        row: dict[str, Any] = {
            "model": label,
            "seeds": len(ms),
            "params (M)": f"{runs[key][0]['params'] / 1e6:.2f}",
        }
        for k, col in COLUMNS.items():
            row[col] = _ms([x[k] for x in ms])
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_table(runs: list[dict[str, Any]], split: str) -> pd.DataFrame:
    frames = [pd.read_csv(m["_dir"] / f"per_class_{split}.csv") for m in runs]
    stacked = pd.concat(frames)
    g = stacked.groupby("class", sort=False)
    out = g[["precision", "recall", "f1"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["support"] = g["support"].first()
    return out.reset_index()


def field_analysis(run: dict[str, Any]) -> dict[str, float]:
    """How a model fails on PlantDoc (one seed)."""
    p, y = _probs(run, "plantdoc")
    pred, conf = p.argmax(1), p.max(1)
    present = set(y.tolist())
    top4 = sum(c for _, c in Counter(pred.tolist()).most_common(4)) / len(y)
    crop = lambda i: split_class(PLANTVILLAGE_CLASSES[i])[0]  # noqa: E731
    healthy_true = np.array([is_healthy(PLANTVILLAGE_CLASSES[i]) for i in y])
    healthy_pred = np.array([is_healthy(PLANTVILLAGE_CLASSES[i]) for i in pred])
    correct = pred == y
    return {
        "accuracy": float(correct.mean()),
        "crop_accuracy": float(np.mean([crop(a) == crop(b) for a, b in zip(pred, y, strict=True)])),
        "healthy_vs_diseased_accuracy": float((healthy_true == healthy_pred).mean()),
        "always_diseased_baseline": float((~healthy_true).mean()),
        "share_top4_predicted_classes": float(top4),
        "share_predicted_absent_classes": float(np.mean([k not in present for k in pred])),
        "mean_conf_correct": float(conf[correct].mean()) if correct.any() else float("nan"),
        "mean_conf_wrong": float(conf[~correct].mean()),
    }


def selective(run: dict[str, Any], split: str) -> list[dict[str, float]]:
    """Coverage and accuracy when answers below a confidence threshold are rejected."""
    p, y = _probs(run, split)
    conf, correct = p.max(1), p.argmax(1) == y
    rows = []
    for t in THRESHOLDS:
        keep = conf >= t
        rows.append(
            {
                "threshold": t,
                "coverage": float(keep.mean()),
                "accuracy_answered": float(correct[keep].mean()) if keep.any() else float("nan"),
            }
        )
    return rows


# ----------------------------------------------------------------------------- figures


def _style(ax: plt.Axes) -> None:
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2)


def fig_generalization(runs: dict[str, list[dict[str, Any]]], out: Path) -> Path:
    keys = [k for k in MAIN if k in runs]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(keys) + 1.4))
    y = np.arange(len(keys))
    for j, (metric, label) in enumerate(
        [("test_acc", "clean test (lab images)"), ("pd_acc", "PlantDoc (field images)")]
    ):
        vals = np.array(
            [[summary_metrics(m)[metric] * 100 for m in runs[k]] for k in keys], dtype=object
        )
        mu = np.array([np.mean(v) for v in vals])
        sd = np.array([np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals])
        ax.errorbar(
            mu,
            y + (0.12 if j == 0 else -0.12),
            xerr=sd,
            fmt="o" if j == 0 else "s",
            color=SERIES[j],
            ms=7,
            capsize=3,
            lw=1.5,
            label=label,
        )
        for yi, m, e in zip(y, mu, sd, strict=True):
            ax.annotate(
                f"{m:.1f}",
                (m + e, yi + (0.12 if j == 0 else -0.12)),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK,
            )
    ax.set_yticks(y, [MAIN[k] for k in keys])
    ax.set_xlim(0, 105)
    ax.set_xlabel("top-1 accuracy (%), mean ± std over seeds")
    ax.set_title("Lab test accuracy vs field accuracy", loc="left", fontsize=11, weight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.28), ncols=2, frameon=False, fontsize=8)
    _style(ax)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "generalization.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_confusion(run: dict[str, Any], split: str, title: str, out: Path) -> Path:
    cm = pd.read_csv(run["_dir"] / f"confusion_{split}.csv", index_col=0).to_numpy(dtype=float)
    rows = cm.sum(axis=1) > 0
    norm = np.divide(
        cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(1, keepdims=True) > 0
    )
    names = [display_name(c) for c in PLANTVILLAGE_CLASSES]
    fig, ax = plt.subplots(figsize=(10, 0.24 * rows.sum() + 2.5))
    im = ax.imshow(norm[rows], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(rows.sum()), [n for n, r in zip(names, rows, strict=True) if r], fontsize=6)
    ax.set_xticks(range(len(names)), names, fontsize=5.5, rotation=90)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, label="share of true class")
    p = out / "confusion" / f"{run['name']}_{split}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_field_predictions(run: dict[str, Any], label: str, out: Path) -> Path:
    p, y = _probs(run, "plantdoc")
    pred = Counter(p.argmax(1).tolist())
    true = Counter(y.tolist())
    top = [k for k, _ in pred.most_common(12)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    yy = np.arange(len(top))[::-1]
    ax.barh(
        yy + 0.2,
        [pred[k] / len(y) * 100 for k in top],
        0.38,
        color=SERIES[0],
        label="share of predictions",
    )
    ax.barh(
        yy - 0.2,
        [true[k] / len(y) * 100 for k in top],
        0.38,
        color=SERIES[1],
        label="share of true labels",
    )
    ax.set_yticks(yy, [display_name(PLANTVILLAGE_CLASSES[k]) for k in top], fontsize=7.5)
    ax.set_xlabel("% of PlantDoc images")
    ax.set_title(
        f"What {label} predicts on field images (seed 42)", loc="left", fontsize=10, weight="bold"
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _style(ax)
    ax.grid(axis="y", visible=False)
    path = out / f"field_predictions_{run['name']}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_selective(run: dict[str, Any], label: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for j, (split, name) in enumerate([("test", "clean test"), ("plantdoc", "PlantDoc")]):
        p, y = _probs(run, split)
        conf, correct = p.max(1), p.argmax(1) == y
        order = np.argsort(-conf)
        cum = np.cumsum(correct[order]) / np.arange(1, len(y) + 1)
        cov = np.arange(1, len(y) + 1) / len(y)
        ax.plot(cov * 100, cum * 100, color=SERIES[j], lw=2, label=name)
    ax.set_xlabel("coverage: % of images answered (most confident first)")
    ax.set_ylabel("accuracy on answered images (%)")
    ax.set_title(
        f"Rejecting low-confidence answers: {label}", loc="left", fontsize=10, weight="bold"
    )
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    path = out / f"selective_{run['name']}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- main


def build(results: Path, out: Path) -> list[Path]:
    runs = load(results)
    out.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    main = results_table(runs, MAIN)
    main.to_csv(out / "main_results.csv", index=False)
    (out / "main_results.md").write_text(_md(main))
    made += [out / "main_results.md"]
    abl = results_table(runs, ABLATIONS)
    if len(abl) > 1:
        abl.to_csv(out / "ablations.csv", index=False)
        (out / "ablations.md").write_text(_md(abl))
        made.append(out / "ablations.md")

    pc_dir = out / "per_class"
    pc_dir.mkdir(exist_ok=True)
    f1_cols = {}
    for key in [*MAIN, *ABLATIONS]:
        if key not in runs:
            continue
        for split in ("test", "plantdoc"):
            t = per_class_table(runs[key], split)
            t.to_csv(pc_dir / f"{key}_{split}.csv", index=False)
            if split == "test" and key in MAIN:
                f1_cols[MAIN[key]] = (t.set_index("class")["f1_mean"] * 100).round(2)
    if f1_cols:
        f1 = pd.DataFrame(f1_cols)
        f1.index = [display_name(c) for c in f1.index]
        f1 = f1.reset_index(names="class")
        (out / "per_class_f1_test.md").write_text(_md(f1))
        made.append(out / "per_class_f1_test.md")

    field_rows, sel_rows = [], []
    for key, label in MAIN.items():
        if key not in runs:
            continue
        seed42 = next((m for m in runs[key] if m["seed"] == 42), runs[key][0])
        fa = [field_analysis(m) for m in runs[key]]
        field_rows.append({"model": label, **{k: _ms([x[k] for x in fa]) for k in fa[0]}})
        for split in ("test", "plantdoc"):
            for r in selective(seed42, split):
                sel_rows.append({"model": label, "set": split, **r})
        made.append(fig_confusion(seed42, "test", f"{label}: clean test (seed 42)", out))
        made.append(fig_confusion(seed42, "plantdoc", f"{label}: PlantDoc (seed 42)", out))
    field = pd.DataFrame(field_rows)
    field.to_csv(out / "field_analysis.csv", index=False)
    (out / "field_analysis.md").write_text(_md(field))
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(out / "selective_prediction.csv", index=False)
    (out / "selective_prediction.md").write_text(_md(sel.round(4)))
    made += [
        out / "field_analysis.md",
        out / "selective_prediction.md",
        fig_generalization(runs, out),
    ]

    best = max(
        (k for k in MAIN if k in runs),
        key=lambda k: np.mean([summary_metrics(m)["test_acc"] for m in runs[k]]),
    )
    best42 = next((m for m in runs[best] if m["seed"] == 42), runs[best][0])
    made += [fig_field_predictions(best42, MAIN[best], out), fig_selective(best42, MAIN[best], out)]
    return made


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("../docs/figures/phase3"))
    args = p.parse_args(argv)
    for f in build(args.results, args.out):
        print("wrote", f)


if __name__ == "__main__":
    main()
