"""Collect run outputs and generate the reproduction tables and figures.

    python -m leafdx.train.report collect outputs        # copy small files into results/
    python -m leafdx.train.report reproduction           # -> docs/figures/reproduction/

``results/`` is committed; it holds everything a reported number is computed from
(config, env, history, metrics, per-class tables) but not model weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MISSING = "\u2013"  # en dash for empty table cells
RESULT_FILES = ["config.yaml", "env.json", "history.csv", "metrics.json"]
RESULT_GLOBS = ["per_class_*.csv", "confusion_*.csv", "probs_*.npz"]

# Ali Usman and Syed Moazam Ali, Table 3 ("Accuracy Measure during Training of Models"), 20 epochs.
PAPER_TABLE3: dict[str, dict[str, float]] = {
    "paper_cnn": {"accuracy": 0.9711, "loss": 0.0898, "val_accuracy": 0.9025, "val_loss": 0.3605},
    "paper_mobilenet": {
        "accuracy": 0.9348,
        "loss": 0.2060,
        "val_accuracy": 0.9547,
        "val_loss": 0.1419,
    },
}
METRIC_LABELS = {
    "accuracy": "train accuracy",
    "loss": "train loss",
    "val_accuracy": "validation accuracy",
    "val_loss": "validation loss",
}
SERIES = ["#2a78d6", "#eb6834"]  # dataviz reference palette, validated pair


def collect(outputs: Path, results: Path) -> list[Path]:
    """Copy the small files of every finished run (one with metrics.json)."""
    copied = []
    for metrics in sorted(outputs.glob("*/seed*/metrics.json")):
        run_dir = metrics.parent
        dest = results / run_dir.parent.name / run_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        names = RESULT_FILES + [p.name for g in RESULT_GLOBS for p in run_dir.glob(g)]
        for name in names:
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, dest / name)
        copied.append(dest)
    return copied


def load_runs(results: Path, prefix: str = "paper_") -> list[dict[str, Any]]:
    runs = []
    for m in sorted(results.glob(f"{prefix}*/seed*/metrics.json")):
        metrics = json.loads(m.read_text())
        history = pd.read_csv(m.parent / "history.csv")
        runs.append({"dir": m.parent, "metrics": metrics, "history": history})
    return runs


def model_key(name: str) -> str:
    return "paper_cnn" if name.startswith("paper_cnn") else "paper_mobilenet"


def schedule_of(name: str) -> str:
    return name.split("_")[-1]


def comparison_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = r["metrics"]
        paper = PAPER_TABLE3[model_key(m["name"])]
        for key, label in METRIC_LABELS.items():
            ours = m["last_epoch"].get(key)
            row = {
                "model": model_key(m["name"]).removeprefix("paper_"),
                "schedule": schedule_of(m["name"]),
                "seed": m["seed"],
                "metric": label + " (last epoch, as Keras reports)",
                "paper": paper[key],
                "reproduced": ours,
                "difference": None if ours is None else ours - paper[key],
            }
            rows.append(row)
        full = m["val_full"]
        rows.append(
            {
                "model": model_key(m["name"]).removeprefix("paper_"),
                "schedule": schedule_of(m["name"]),
                "seed": m["seed"],
                "metric": "validation accuracy, full 17,572 images (argmax)",
                "paper": 0.95 if model_key(m["name"]) == "paper_mobilenet" else None,
                "reproduced": full["accuracy"],
                "difference": None,
            }
        )
        rows.append(
            {
                "model": model_key(m["name"]).removeprefix("paper_"),
                "schedule": schedule_of(m["name"]),
                "seed": m["seed"],
                "metric": "validation macro-F1, full set",
                "paper": 0.95 if model_key(m["name"]) == "paper_mobilenet" else None,
                "reproduced": full["macro_f1"],
                "difference": None,
            }
        )
    return pd.DataFrame(rows)


def _fmt(v: Any, pct: bool) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return MISSING
    return f"{v * 100:.2f}" if pct else f"{v:.4f}"


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| model | schedule | seed | metric | paper | reproduced | difference |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples(index=False):
        pct = "loss" not in r.metric
        diff = _fmt(r.difference, pct)
        if diff != MISSING and not diff.startswith("-"):
            diff = "+" + diff
        unit = " pp" if pct and diff != MISSING else ""
        lines.append(
            f"| {r.model} | {r.schedule} | {r.seed} | {r.metric} | {_fmt(r.paper, pct)} | "
            f"{_fmt(r.reproduced, pct)} | {diff}{unit} |"
        )
    return "\n".join(lines) + "\n"


def curves_figure(runs: list[dict[str, Any]], out: Path) -> Path:
    keys = sorted(
        {(model_key(r["metrics"]["name"]), schedule_of(r["metrics"]["name"])) for r in runs}
    )
    fig, axes = plt.subplots(
        1, len(keys), figsize=(3.6 * len(keys), 3.2), squeeze=False, sharey=True
    )
    for ax, key in zip(axes[0], keys, strict=True):
        for r in runs:
            name = r["metrics"]["name"]
            if (model_key(name), schedule_of(name)) != key:
                continue
            h = r["history"]
            ep = h["epoch"] + 1
            ax.plot(ep, h["accuracy"], color=SERIES[0], lw=2, label="train")
            ax.plot(ep, h["val_accuracy"], color=SERIES[1], lw=2, label="validation")
        p = PAPER_TABLE3[key[0]]
        ax.scatter(
            [20, 20],
            [p["accuracy"], p["val_accuracy"]],
            marker="x",
            s=40,
            color=["#0b0b0b", "#52514e"],
            zorder=3,
            label="paper (Table 3)",
        )
        ax.set_title(f"{key[0].removeprefix('paper_')}, {key[1]}", loc="left", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].set_ylabel("accuracy (as Keras reports)")
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=8)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "curves.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def reproduction(results: Path, out: Path) -> list[Path]:
    runs = load_runs(results)
    if not runs:
        raise SystemExit(f"no paper_* runs in {results}")
    out.mkdir(parents=True, exist_ok=True)
    table = comparison_table(runs)
    table.to_csv(out / "paper_vs_reproduced.csv", index=False)
    (out / "paper_vs_reproduced.md").write_text(to_markdown(table))
    runs_tbl = pd.DataFrame(
        [
            {
                "run": r["metrics"]["name"],
                "seed": r["metrics"]["seed"],
                "params": r["metrics"]["params"],
                "trainable_params": r["metrics"]["trainable_params"],
                "images_seen": r["metrics"]["images_seen"],
                "train_minutes": round(r["metrics"]["train_seconds"] / 60, 1),
                "device": ", ".join(json.loads((r["dir"] / "env.json").read_text())["gpus"])
                or "CPU",
            }
            for r in runs
        ]
    )
    runs_tbl.to_csv(out / "runs.csv", index=False)
    return [out / "paper_vs_reproduced.md", out / "runs.csv", curves_figure(runs, out)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("outputs", type=Path)
    c.add_argument("--results", type=Path, default=Path("results"))
    r = sub.add_parser("reproduction")
    r.add_argument("--results", type=Path, default=Path("results"))
    r.add_argument("--out", type=Path, default=Path("../docs/figures/reproduction"))
    args = p.parse_args(argv)
    if args.cmd == "collect":
        for d in collect(args.outputs, args.results):
            print("collected", d)
    else:
        for f in reproduction(args.results, args.out):
            print("wrote", f)


if __name__ == "__main__":
    main()
