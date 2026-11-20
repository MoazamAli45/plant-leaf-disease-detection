"""Phase 6 tables and figures from results/p6_*/ (protocol: docs/field.md).

python -m leafdx.field.report        # -> ../docs/figures/phase6/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = {
    "p3_mobilenet_v1": "MobileNetV1",
    "p3_efficientnet_b0": "EfficientNetB0",
    "p3_mobilenet_v2": "MobileNetV2",
    "p3_mobilenet_v3_small": "MobileNetV3-Small",
    "p3_cnn_softmax": "Custom CNN",
}
METHOD_NAMES = {
    "base": "baseline",
    "tta": "test-time augmentation",
    "adabn": "AdaBN (unlabelled field images)",
    "ft": "fine-tuning",
    "fttta": "fine-tuning + test-time augmentation",
}
SETS = ("field_val", "field_test", "lab_check")
COLS = {
    "accuracy": "acc",
    "macro_f1": "macro_f1",
    "crop_accuracy": "crop_acc",
    "ece": "ece",
    "restricted_accuracy": "acc_28",
}


def collect(results: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(results.glob("p6_*/seed*/metrics.json")):
        m = json.loads(f.read_text())
        row = {
            "run": f.parent.parent.name,
            "seed": int(f.parent.name[4:]),
            "method": m["method"],
            "model": m["model"],
            "k": m.get("k"),
            "lr": m.get("lr"),
            "best_epoch": m.get("best_epoch"),
        }
        if m["method"] in ("ft", "fttta") and m.get("k") is None:
            row["k"] = "all"
        for s in SETS:
            for key, short in COLS.items():
                row[f"{s}_{short}"] = m["metrics"][s][key] * 100
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["run", "method", "model", "k", "lr"]
    metric_cols = [c for c in df.columns if c.startswith(SETS)]
    g = df.fillna({"k": "-", "lr": "-"}).groupby(keys, sort=False)
    out = g[metric_cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["seeds"] = g.size()
    return out.reset_index()


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.2f}" if np.isnan(std) else f"{mean:.2f} ± {std:.2f}"


def table_md(s: pd.DataFrame) -> str:
    head = [
        "Model",
        "Method",
        "k / class",
        "Seeds",
        "Field-val acc.",
        "Field-test acc.",
        "Field-test macro-F1",
        "Field-test crop acc.",
        "Field-test ECE",
        "Lab check acc.",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    order = {"base": 0, "tta": 1, "adabn": 2, "ft": 3, "fttta": 4}
    s = s.assign(
        _m=s["method"].map(order),
        _k=s["k"].map(lambda k: 999 if k == "all" else (k if isinstance(k, int | float) else -1)),
    )
    for _, r in s.sort_values(["model", "_m", "_k", "run"]).iterrows():
        name = METHOD_NAMES[r["method"]]
        if r["method"] in ("ft", "fttta") and r["lr"] not in ("-", 3e-5):
            name += f" (lr {r['lr']:g})"
        if "_replay" in r["run"]:
            name += f" ({r['run'].rsplit('_replay', 1)[1]} lab images/class replayed)"
        lines.append(
            "| "
            + " | ".join(
                [
                    MODELS.get(r["model"], r["model"]),
                    name,
                    str(r["k"]),
                    str(r["seeds"]),
                    _fmt(r["field_val_acc_mean"], r["field_val_acc_std"]),
                    _fmt(r["field_test_acc_mean"], r["field_test_acc_std"]),
                    _fmt(r["field_test_macro_f1_mean"], r["field_test_macro_f1_std"]),
                    _fmt(r["field_test_crop_acc_mean"], r["field_test_crop_acc_std"]),
                    _fmt(r["field_test_ece_mean"], r["field_test_ece_std"]),
                    _fmt(r["lab_check_acc_mean"], r["lab_check_acc_std"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def learning_curve(df: pd.DataFrame, out: Path, metric: str = "field_test_acc") -> Path | None:
    # main sweep only: default learning rate and replay size
    ft = df[
        (df["method"] == "ft")
        & (df["lr"].fillna(3e-5) == 3e-5)
        & ~df["run"].str.contains("_replay")
    ]
    if ft.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, (model, sub) in enumerate(ft.groupby("model")):
        colour = f"C{i}"
        ks = sorted(sub["k"].unique(), key=lambda k: 10**6 if k == "all" else k)
        # "all" is plotted at the median number of field-train images per class (45)
        xs = [45 if k == "all" else k for k in ks]
        g = sub.groupby("k")[metric]
        means = [g.mean()[k] for k in ks]
        stds = [g.std().fillna(0)[k] for k in ks]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            color=colour,
            label=f"{MODELS.get(model, model)}, fine-tuned",
        )
        for method, style in (("base", ":"), ("tta", "--")):
            b = df[(df["method"] == method) & (df["model"] == model)][metric]
            if len(b):
                ax.axhline(
                    b.mean(),
                    color=colour,
                    linestyle=style,
                    linewidth=1,
                    label=f"{MODELS.get(model, model)}, {METHOD_NAMES[method]}",
                )
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 45], ["5", "10", "20", "all (≈45)"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("labelled field images per class")
    ax.set_ylabel("field-test accuracy (%)" if metric == "field_test_acc" else metric)
    ax.set_title("Field accuracy vs. number of labelled field images")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = out / "learning_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def per_class_deployed(results: Path, out: Path) -> pd.DataFrame | None:
    """Field-test accuracy per class, before (Phase 3 baseline) and after (the deployed run,
    with test-time augmentation if the app uses it), from the saved predictions."""
    from leafdx.data.classes import PLANTVILLAGE_CLASSES, display_name

    dep = out / "deployed.json"
    if not dep.exists():
        return None
    d = json.loads(dep.read_text())
    run = Path(d["run"])
    after_dir = results / (
        run.parent.name.replace("p6_ft_", "p6_fttta_", 1) if d.get("tta") else run.parent.name
    )
    after = np.load(after_dir / run.name / "probs_field_test.npz")
    before = np.load(results / f"p6_base_{d['base_model']}" / "seed0" / "probs_field_test.npz")
    assert (after["paths"] == before["paths"]).all()
    y = after["y"]
    rows = []
    for c in np.unique(y):
        m = y == c
        rows.append(
            {
                "class": display_name(PLANTVILLAGE_CLASSES[c]),
                "n_field_test": int(m.sum()),
                "before_acc": float((before["probs"][m].argmax(1) == c).mean() * 100),
                "after_acc": float((after["probs"][m].argmax(1) == c).mean() * 100),
            }
        )
    df = pd.DataFrame(rows).sort_values("after_acc", ascending=False)
    df.to_csv(out / "per_class_deployed.csv", index=False, float_format="%.1f")
    lines = ["| Class | Field-test images | Before (%) | After (%) |", "|---|---|---|---|"]
    lines += [
        f"| {r['class']} | {r['n_field_test']} | {r['before_acc']:.1f} | {r['after_acc']:.1f} |"
        for _, r in df.iterrows()
    ]
    (out / "per_class_deployed.md").write_text("\n".join(lines) + "\n")
    return df


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("../docs/figures/phase6"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    df = collect(a.results)
    df.to_csv(a.out / "runs.csv", index=False)
    s = summarise(df)
    s.to_csv(a.out / "results.csv", index=False)
    (a.out / "results.md").write_text(table_md(s))
    fig = learning_curve(df, a.out)
    per_class_deployed(a.results, a.out)
    print(table_md(s))
    print("figure:", fig)


if __name__ == "__main__":
    main()
