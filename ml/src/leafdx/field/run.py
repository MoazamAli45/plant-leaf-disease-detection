"""Run Phase 6 experiments (protocol: docs/field.md).

    python -m leafdx.field.run zero --model p3_mobilenet_v1                # B, TTA, AdaBN
    python -m leafdx.field.run finetune --model p3_mobilenet_v1 --k 10 --seed 0
    python -m leafdx.field.run finetune --model p3_mobilenet_v1 --seed 0   # k = all
    python -m leafdx.field.run tta --run results/p6_ft_p3_mobilenet_v1_kall/seed0  # FT + TTA

Every run evaluates on field-val, field-test and the lab check set and writes
results/p6_<method>_<model>[...]/seed<s>/{metrics.json, probs_<set>.npz}. Decisions are made on
field-val only (docs/field.md §5).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from leafdx.data.classes import PLANTVILLAGE_CLASSES, split_class
from leafdx.field import data as fd
from leafdx.field.methods import adabn, batchnorm_layers, finetune, load_model, predict
from leafdx.train.metrics import classification_metrics


def evaluate(y: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    summary, _, _ = classification_metrics(y, probs, PLANTVILLAGE_CLASSES)
    pred = probs.argmax(axis=1)
    crop = [split_class(PLANTVILLAGE_CLASSES[i])[0] for i in range(len(PLANTVILLAGE_CLASSES))]
    summary["crop_accuracy"] = float(
        np.mean([crop[a] == crop[b] for a, b in zip(pred, y, strict=True)])
    )
    return summary


def _env() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "platform": platform.platform(),
        "devices": [d.device_type for d in tf.config.list_physical_devices()],
    }


def _save(
    out: Path, info: dict[str, Any], sets: dict[str, fd.ImageSet], probs: dict[str, np.ndarray]
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    info["metrics"] = {}
    for name, s in sets.items():
        info["metrics"][name] = evaluate(s.y, probs[name])
        np.savez_compressed(
            out / f"probs_{name}.npz", probs=probs[name], y=s.y, paths=np.array(s.paths)
        )
    info["env"] = _env()
    (out / "metrics.json").write_text(json.dumps(info, indent=1) + "\n")
    short = {k: round(v["accuracy"] * 100, 2) for k, v in info["metrics"].items()}
    print(out, short)
    return info


def eval_sets(a: argparse.Namespace) -> dict[str, fd.ImageSet]:
    return {
        "field_val": fd.field("val", a.data_root, a.cache),
        "field_test": fd.field("test", a.data_root, a.cache),
        "lab_check": fd.lab_check(a.data_root, a.cache),
    }


def run_zero(a: argparse.Namespace) -> None:
    sets = eval_sets(a)
    path = str(a.models / "_export" / a.model / "inference.keras")
    model = load_model(path)
    for method, tta in (("base", False), ("tta", True)):
        out = a.results / f"p6_{method}_{a.model}" / "seed0"
        if out.exists() and not a.overwrite:
            continue
        t = time.time()
        probs = {n: predict(model, s, tta=tta) for n, s in sets.items()}
        _save(out, {"method": method, "model": a.model, "seconds": time.time() - t}, sets, probs)
    if not batchnorm_layers(model):
        print(a.model, "has no batch normalisation: AdaBN not applicable")
        return
    unlabelled = fd.field("train", a.data_root, a.cache)
    for seed in a.seeds:
        out = a.results / f"p6_adabn_{a.model}" / f"seed{seed}"
        if out.exists() and not a.overwrite:
            continue
        m = load_model(path)
        n_bn = adabn(m, unlabelled, seed=seed)
        probs = {n: predict(m, s) for n, s in sets.items()}
        info = {
            "method": "adabn",
            "model": a.model,
            "seed": seed,
            "n_bn_layers": n_bn,
            "n_unlabelled": len(unlabelled.x),
        }
        _save(out, info, sets, probs)


def run_finetune(a: argparse.Namespace) -> None:
    k_tag = f"k{a.k}" if a.k is not None else "kall"
    lr_tag = "" if a.lr == 3e-5 else f"_lr{a.lr:g}"
    replay_tag = "" if a.replay == 20 else f"_replay{a.replay}"
    out = a.results / f"p6_ft_{a.model}_{k_tag}{lr_tag}{replay_tag}" / f"seed{a.seed}"
    if out.exists() and not a.overwrite:
        print("exists:", out)
        return
    sets = eval_sets(a)
    train_field = fd.k_shot(fd.field("train", a.data_root, a.cache), a.k, a.seed)
    replay = fd.lab_replay(a.data_root, a.cache, a.replay, a.seed)
    model = load_model(str(a.models / "_export" / a.model / "inference.keras"))
    t = time.time()
    fit = finetune(
        model, [train_field, replay], sets["field_val"], a.lr, a.epochs, a.patience, a.seed
    )
    probs = {n: predict(model, s) for n, s in sets.items()}
    info = {
        "method": "ft",
        "model": a.model,
        "k": a.k,
        "seed": a.seed,
        "lr": a.lr,
        "n_field_train": len(train_field.x),
        "n_lab_replay": len(replay.x),
        "seconds": time.time() - t,
        **fit,
    }
    _save(out, info, sets, probs)
    if a.save_model:
        model.save(out / "model.keras")


def run_tta(a: argparse.Namespace) -> None:
    """Test-time augmentation on a saved fine-tuned model (FT + TTA, docs/field.md §4)."""
    src = json.loads((a.run / "metrics.json").read_text())
    out = a.results / a.run.parent.name.replace("p6_ft_", "p6_fttta_", 1) / a.run.name
    if out.exists() and not a.overwrite:
        print("exists:", out)
        return
    sets = eval_sets(a)
    model = keras.models.load_model(a.run / "model.keras", compile=False)
    t = time.time()
    probs = {n: predict(model, s, tta=True) for n, s in sets.items()}
    keep = ("model", "k", "seed", "lr", "n_field_train", "n_lab_replay", "best_epoch")
    info = {"method": "fttta", "source_run": str(a.run), **{k: src.get(k) for k in keep}}
    info["seconds"] = time.time() - t
    _save(out, info, sets, probs)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("zero", "finetune"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--models", type=Path, default=Path("models"))
        s.add_argument("--results", type=Path, default=Path("results"))
        s.add_argument("--data-root", type=Path, default=Path("data"))
        s.add_argument("--cache", type=Path, default=Path("data/cache/p6"))
        s.add_argument("--overwrite", action="store_true")
    sub.choices["zero"].add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ft = sub.choices["finetune"]
    ft.add_argument("--k", type=int, default=None, help="field images per class (default: all)")
    ft.add_argument("--seed", type=int, default=0)
    ft.add_argument("--lr", type=float, default=3e-5)
    ft.add_argument("--epochs", type=int, default=15)
    ft.add_argument("--patience", type=int, default=3)
    ft.add_argument("--replay", type=int, default=20, help="lab images per class mixed in")
    ft.add_argument("--save-model", action="store_true")
    tt = sub.add_parser("tta")
    tt.add_argument("--run", type=Path, required=True, help="a fine-tuning run with model.keras")
    for flag, default in (
        ("--results", "results"),
        ("--data-root", "data"),
        ("--cache", "data/cache/p6"),
    ):
        tt.add_argument(flag, type=Path, default=Path(default))
    tt.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    {"zero": run_zero, "finetune": run_finetune, "tta": run_tta}[a.cmd](a)


if __name__ == "__main__":
    main()
