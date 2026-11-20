"""Phase 7: more field data, EfficientNetB0, an independent test source (docs/field2.md).

    python -m leafdx.field.phase7 reference                       # deployed Phase 6 model
    python -m leafdx.field.phase7 run --model p3_mobilenet_v1 --method PF --seed 0
    python -m leafdx.field.phase7 report                          # -> ../docs/figures/phase7/

Every run evaluates on PlantDoc val/test, FieldPlant val/test, CD&S and the lab check set, with
and without test-time augmentation, and writes results/p7_<method>_<model>/seed<s>/
{metrics.json, probs_<set>[_tta].npz, model.keras}. Choices use the validation sets only.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd

from leafdx.data.classes import PLANTVILLAGE_CLASSES
from leafdx.field import data as fd
from leafdx.field.methods import finetune, load_model, predict
from leafdx.field.run import _env, evaluate

FIELDPLANT = Path("manifests/fieldplant")
CDS = Path("manifests/cds/all.csv")
METHODS = {
    "P": {"fieldplant_cap": None, "epochs": 15, "patience": 3},
    "PF": {"fieldplant_cap": 100, "epochs": 15, "patience": 3},
    "PF-long": {"fieldplant_cap": 100, "epochs": 30, "patience": 5},
}
REPLAY = 50
LR = 3e-5
EVAL = ("pd_val", "pd_test", "fp_val", "fp_test", "cds", "lab_check")


def fieldplant(split: str, data_root: Path, cache: Path) -> fd.ImageSet:
    return fd.load(f"fp_{split}", pd.read_csv(FIELDPLANT / f"{split}.csv"), data_root, cache)


def cds(data_root: Path, cache: Path) -> fd.ImageSet:
    return fd.load("cds", pd.read_csv(CDS), data_root, cache)


def eval_sets(data_root: Path, cache: Path) -> dict[str, fd.ImageSet]:
    """CD&S is added later (``add-cds``) when its download is not complete yet."""
    sets = {
        "pd_val": fd.field("val", data_root, cache),
        "pd_test": fd.field("test", data_root, cache),
        "fp_val": fieldplant("val", data_root, cache),
        "fp_test": fieldplant("test", data_root, cache),
        "lab_check": fd.lab_check(data_root, cache),
    }
    if CDS.exists():
        sets["cds"] = cds(data_root, cache)
    return sets


def concat(name: str, sets: list[fd.ImageSet]) -> fd.ImageSet:
    return fd.ImageSet(
        name,
        np.concatenate([s.x for s in sets]),
        np.concatenate([s.y for s in sets]),
        [p for s in sets for p in s.paths],
    )


def balanced_accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    pred = probs.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def corn_accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    """Share of photos given any corn class (the crop is right, whatever the disease)."""
    corn = [i for i, c in enumerate(PLANTVILLAGE_CLASSES) if c.startswith("Corn")]
    return float(np.isin(probs.argmax(1), corn).mean())


def metrics(sets: dict[str, fd.ImageSet], probs: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, p in probs.items():
        s = sets[name.removesuffix("_tta")]
        m = evaluate(s.y, p)
        m["balanced_accuracy"] = balanced_accuracy(s.y, p)
        if name.startswith("cds"):
            m["corn_accuracy"] = corn_accuracy(s.y, p)
        out[name] = m
    for tta in ("", "_tta"):
        out[f"selection{tta}"] = (
            out[f"pd_val{tta}"]["accuracy"] + out[f"fp_val{tta}"]["balanced_accuracy"]
        ) / 2
    return out


def evaluate_model(
    model: keras.Model, sets: dict[str, fd.ImageSet], out: Path, info: dict[str, Any]
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    probs: dict[str, np.ndarray] = {}
    for name, s in sets.items():
        for tta in (False, True):
            key = name + ("_tta" if tta else "")
            probs[key] = predict(model, s, tta=tta)
            np.savez_compressed(out / f"probs_{key}.npz", probs=probs[key], y=s.y)
    info["metrics"] = metrics(sets, probs)
    info["env"] = _env()
    (out / "metrics.json").write_text(json.dumps(info, indent=1) + "\n")
    m = info["metrics"]
    keys = ("pd_test_tta", "fp_test_tta", "cds_tta")
    short = {k: round(m[k]["accuracy"] * 100, 2) for k in keys if k in m}
    short["selection_tta"] = round(m["selection_tta"] * 100, 2)
    short["lab_check"] = round(m["lab_check"]["accuracy"] * 100, 2)
    print(out, short, flush=True)
    return info


def run(a: argparse.Namespace) -> None:
    cfg = METHODS[a.method]
    out = a.results / f"p7_{a.method}_{a.model}" / f"seed{a.seed}"
    if (out / "metrics.json").exists() and not a.overwrite:
        print("exists:", out)
        return
    sets = eval_sets(a.data_root, a.cache)
    train = [fd.field("train", a.data_root, a.cache)]
    if cfg["fieldplant_cap"]:
        train.append(fd.k_shot(fieldplant("train", a.data_root, a.cache), cfg["fieldplant_cap"], 0))
    replay = fd.lab_replay(a.data_root, a.cache, REPLAY, a.seed)
    val = concat("pd_fp_val", [sets["pd_val"], sets["fp_val"]])
    model = load_model(str(a.models / "_export" / a.model / "inference.keras"))
    t = time.time()
    fit = finetune(model, [*train, replay], val, LR, cfg["epochs"], cfg["patience"], a.seed)
    info = {
        "phase": 7,
        "method": a.method,
        "model": a.model,
        "seed": a.seed,
        "lr": LR,
        "n_field_train": {s.name: len(s.x) for s in train},
        "n_lab_replay": len(replay.x),
        "train_seconds": time.time() - t,
        **cfg,
        **fit,
    }
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    evaluate_model(model, sets, out, info)


def reference(a: argparse.Namespace) -> None:
    """The deployed Phase 6 model and the Phase 3 starting models on the Phase 7 test sets."""
    sets = eval_sets(a.data_root, a.cache)
    runs = {
        "p7_ref_p6deployed": Path("results/p6_ft_p3_mobilenet_v1_kall_replay50/seed0/model.keras"),
        "p7_ref_p3_mobilenet_v1": a.models / "_export" / "p3_mobilenet_v1" / "inference.keras",
        "p7_ref_p3_efficientnet_b0": a.models
        / "_export"
        / "p3_efficientnet_b0"
        / "inference.keras",
    }
    for name, path in runs.items():
        out = a.results / name / "seed0"
        if (out / "metrics.json").exists() and not a.overwrite:
            print("exists:", out)
            continue
        model = keras.models.load_model(path, compile=False)
        evaluate_model(model, sets, out, {"phase": 7, "method": "reference", "source": str(path)})


def add_cds(a: argparse.Namespace) -> None:
    """Evaluate every saved Phase 7 model on CD&S (once its download is complete)."""
    s = cds(a.data_root, a.cache)
    for f in sorted(a.results.glob("p7_*/seed*/metrics.json")):
        info = json.loads(f.read_text())
        if "cds" in info["metrics"] and not a.overwrite:
            continue
        src = f.parent / "model.keras" if (f.parent / "model.keras").exists() else info["source"]
        model = keras.models.load_model(src, compile=False)
        for tta in (False, True):
            key = "cds" + ("_tta" if tta else "")
            p = predict(model, s, tta=tta)
            np.savez_compressed(f.parent / f"probs_{key}.npz", probs=p, y=s.y)
            m = evaluate(s.y, p)
            m["balanced_accuracy"] = balanced_accuracy(s.y, p)
            m["corn_accuracy"] = corn_accuracy(s.y, p)
            info["metrics"][key] = m
        f.write_text(json.dumps(info, indent=1) + "\n")
        print(f.parent, round(info["metrics"]["cds_tta"]["accuracy"] * 100, 2), flush=True)


# ------------------------------------------------------------------------------ diagnostics


def bootstrap_ci(
    y: np.ndarray,
    probs: np.ndarray,
    stat: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile bootstrap interval of a test statistic (accuracy by default), resampling
    test images with replacement."""
    stat = stat or (lambda yy, pp: float((pp.argmax(1) == yy).mean()))
    rng = np.random.default_rng(seed)
    vals = [stat(y[i], probs[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def diagnose(a: argparse.Namespace) -> None:
    """Overfitting checks for every fine-tuned run (docs/field2.md §7, item 1):
    accuracy on its own field training images (no augmentation) against validation, the
    epoch of the best validation loss, and 95% bootstrap intervals of the test scores."""
    pd_train = fd.field("train", a.data_root, a.cache)
    fp_train = fd.k_shot(fieldplant("train", a.data_root, a.cache), 100, 0)
    rows = []
    for f in sorted(a.results.glob("p7_*/seed*/metrics.json")):
        info = json.loads(f.read_text())
        m = info["metrics"]
        row: dict[str, Any] = {
            "run": f.parent.parent.name,
            "seed": int(f.parent.name[4:]),
            "method": info["method"],
            "model": info.get("model", f.parent.parent.name.removeprefix("p7_ref_")),
        }
        if info["method"] != "reference":
            model = keras.models.load_model(f.parent / "model.keras", compile=False)
            row["pd_train_acc"] = float((predict(model, pd_train).argmax(1) == pd_train.y).mean())
            row["gap_pd"] = row["pd_train_acc"] - m["pd_val"]["accuracy"]
            if info["method"] != "P":
                row["fp_train_bal"] = balanced_accuracy(fp_train.y, predict(model, fp_train))
                row["gap_fp"] = row["fp_train_bal"] - m["fp_val"]["balanced_accuracy"]
            h = info["history"]
            row["best_epoch"] = info["best_epoch"]
            row["epochs_run"] = info["epochs_run"]
            row["final_train_loss"] = h["loss"][-1]
            row["best_val_loss"] = min(h["val_loss"])
        for name, stat in (
            ("pd_test_tta", None),
            ("fp_test_tta", None),
            ("fp_test_tta_bal", balanced_accuracy),
            ("cds_tta", None),
        ):
            file = f.parent / f"probs_{name.removesuffix('_bal')}.npz"
            if not file.exists():
                continue
            z = np.load(file)
            lo, hi = bootstrap_ci(z["y"], z["probs"], stat)
            row[f"{name}_ci_lo"], row[f"{name}_ci_hi"] = lo, hi
        rows.append(row)
        print(row["run"], row["seed"], {k: round(v, 3) for k, v in row.items() if "gap" in k})
    out = Path("../docs/figures/phase7")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "diagnostics.csv", index=False, float_format="%.4f")
    loss_curves(a.results, out)


def loss_curves(results: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted(results.glob("p7_P*_p3_*/seed*/metrics.json"))
    groups: dict[str, list[dict]] = {}
    for f in runs:
        groups.setdefault(f.parent.parent.name, []).append(json.loads(f.read_text()))
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3), sharey=True)
    for ax, (name, infos) in zip(np.atleast_1d(axes), sorted(groups.items()), strict=True):
        for i, info in enumerate(infos):
            h = info["history"]
            ep = np.arange(1, len(h["loss"]) + 1)
            ax.plot(ep, h["loss"], color=f"C{i}", linestyle="--", linewidth=1)
            ax.plot(ep, h["val_loss"], color=f"C{i}", linewidth=1.5, label=f"seed {info['seed']}")
            ax.axvline(info["best_epoch"], color=f"C{i}", linewidth=0.6, alpha=0.5)
        ax.set_title(name.removeprefix("p7_"), fontsize=8)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("loss (dashed: train, solid: validation)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------- report

SETS_REPORTED = {
    "pd_val": ["accuracy"],
    "fp_val": ["balanced_accuracy"],
    "pd_test": ["accuracy", "macro_f1", "ece"],
    "fp_test": ["accuracy", "balanced_accuracy"],
    "cds": ["accuracy", "corn_accuracy"],
    "lab_check": ["accuracy"],
}


def collect(results: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(results.glob("p7_*/seed*/metrics.json")):
        m = json.loads(f.read_text())
        for tta in ("", "_tta"):
            row = {
                "run": f.parent.parent.name,
                "seed": int(f.parent.name[4:]),
                "method": m["method"],
                "model": m.get("model", f.parent.parent.name.removeprefix("p7_ref_")),
                "tta": bool(tta),
                "best_epoch": m.get("best_epoch"),
                "selection": m["metrics"][f"selection{tta}"] * 100,
            }
            for s, keys in SETS_REPORTED.items():
                name = s if s == "lab_check" else s + tta
                for k in keys:
                    v = m["metrics"].get(name, {}).get(k)
                    row[f"{s}_{k}"] = np.nan if v is None else v * 100
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c for c in df.columns if c == "selection" or c.split("_")[0] in ("pd", "fp", "cds", "lab")
    ]
    g = df.groupby(["run", "method", "model", "tta"], sort=False)
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["seeds"] = g.size()
    return out.reset_index()


def table_md(s: pd.DataFrame) -> str:
    head = [
        ("Run", None),
        ("TTA", None),
        ("Seeds", None),
        ("Selection (val)", "selection"),
        ("PlantDoc test", "pd_test_accuracy"),
        ("PlantDoc test macro-F1", "pd_test_macro_f1"),
        ("PlantDoc test ECE", "pd_test_ece"),
        ("FieldPlant test", "fp_test_accuracy"),
        ("FieldPlant test balanced", "fp_test_balanced_accuracy"),
        ("CD&S", "cds_accuracy"),
        ("CD&S corn", "cds_corn_accuracy"),
        ("Lab check", "lab_check_accuracy"),
    ]
    lines = ["| " + " | ".join(h for h, _ in head) + " |", "|" + "---|" * len(head)]

    def f(r: pd.Series, c: str) -> str:
        m, sd = r[f"{c}_mean"], r[f"{c}_std"]
        return f"{m:.2f}" if np.isnan(sd) else f"{m:.2f} ± {sd:.2f}"

    for _, r in s.sort_values(["model", "method", "tta"]).iterrows():
        cells = [r["run"], "yes" if r["tta"] else "no", str(r["seeds"])]
        cells += [f(r, c) for _, c in head[3:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def report(a: argparse.Namespace) -> None:
    out = Path("../docs/figures/phase7")
    out.mkdir(parents=True, exist_ok=True)
    df = collect(a.results)
    df.to_csv(out / "runs.csv", index=False)
    s = summarise(df)
    s.to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(table_md(s))
    print(table_md(s))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "reference", "report", "add-cds", "diagnose"):
        s = sub.add_parser(name)
        s.add_argument("--models", type=Path, default=Path("models"))
        s.add_argument("--results", type=Path, default=Path("results"))
        s.add_argument("--data-root", type=Path, default=Path("data"))
        s.add_argument("--cache", type=Path, default=Path("data/cache/p6"))
        s.add_argument("--overwrite", action="store_true")
    r = sub.choices["run"]
    r.add_argument("--model", required=True)
    r.add_argument("--method", choices=list(METHODS), required=True)
    r.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    cmds = {
        "run": run,
        "reference": reference,
        "report": report,
        "add-cds": add_cds,
        "diagnose": diagnose,
    }
    cmds[a.cmd](a)


if __name__ == "__main__":
    main()
