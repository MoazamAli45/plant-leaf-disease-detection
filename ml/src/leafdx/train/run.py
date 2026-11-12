"""Train one experiment and evaluate it.

    python -m leafdx.train.run --config configs/paper/cnn_steps100.yaml
    python -m leafdx.train.run --config configs/paper/cnn_steps100.yaml --set seed=1 train.epochs=2

Writes ``<output_dir>/<name>/seed<seed>/``:
    config.yaml        resolved configuration
    env.json           library versions, device, git commit, manifest checksums
    history.csv        per-epoch metrics exactly as Keras reported them
    metrics.json       last-epoch Keras numbers (comparable to the paper) + full evaluations
    per_class_<split>.csv, confusion_<split>.csv, probs_<split>.npz
    model.keras        final weights (not committed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from leafdx.data.classes import PLANTVILLAGE_CLASSES
from leafdx.train.config import ExperimentCfg, load_config
from leafdx.train.data import HAS_RANDOM_SHEAR, build_augmentation, load_manifest, make_dataset
from leafdx.train.metrics import classification_metrics
from leafdx.train.models import build_model

log = logging.getLogger(__name__)


class EpochTimer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self._t = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is not None:
            logs["epoch_seconds"] = time.perf_counter() - self._t


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: str | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(cfg: ExperimentCfg) -> dict[str, Any]:
    gpus = tf.config.list_physical_devices("GPU")
    details = [tf.config.experimental.get_device_details(g).get("device_name") for g in gpus]
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "platform": platform.platform(),
        "gpus": details,
        "git_commit": _git_commit(),
        "has_random_shear": HAS_RANDOM_SHEAR,
        "manifest_sha256": {
            "train": _sha256(cfg.data.train_manifest),
            "val": _sha256(cfg.data.val_manifest),
            "test": _sha256(cfg.data.test_manifest),
        },
    }


def _check_files(paths: list[str], name: str) -> None:
    missing = [p for p in paths[:: max(1, len(paths) // 20)] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} sampled files missing, e.g. {missing[0]}. "
            "Set data.data_root / data.npd_root."
        )


def compile_model(model: keras.Model, cfg: ExperimentCfg) -> None:
    t = cfg.train
    opt = keras.optimizers.get(
        {"class_name": t.optimizer, "config": {"learning_rate": t.learning_rate}}
    )
    metrics: list[Any] = ["accuracy"]
    if cfg.model.output_activation == "sigmoid":
        # what Keras would report as "accuracy" had binary cross-entropy been used
        metrics.append(keras.metrics.BinaryAccuracy(name="binary_accuracy"))
    model.compile(optimizer=opt, loss=t.loss, metrics=metrics)


def evaluate_split(
    model: keras.Model, cfg: ExperimentCfg, manifest: str, split: str, run_dir: Path
) -> dict[str, Any]:
    paths, labels = load_manifest(manifest, cfg.data)
    _check_files(paths, split)
    ds = make_dataset(
        paths,
        labels,
        cfg.data,
        cfg.train.batch_size,
        cfg.model.num_classes,
        shuffle=False,
        repeat=False,
        seed=cfg.seed,
    )
    keras_eval = model.evaluate(ds, return_dict=True, verbose=0)
    probs = model.predict(ds, verbose=0)
    summary, per_class, cm = classification_metrics(labels, probs, PLANTVILLAGE_CLASSES)
    per_class.to_csv(run_dir / f"per_class_{split}.csv", index=False)
    pd.DataFrame(cm, index=PLANTVILLAGE_CLASSES, columns=PLANTVILLAGE_CLASSES).to_csv(
        run_dir / f"confusion_{split}.csv"
    )
    np.savez_compressed(
        run_dir / f"probs_{split}.npz", probs=probs.astype(np.float16), labels=labels
    )
    return {"keras": {k: float(v) for k, v in keras_eval.items()}, **summary}


def run(cfg: ExperimentCfg) -> dict[str, Any]:
    run_dir = Path(cfg.output_dir) / cfg.name / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    env = environment(cfg)
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    keras.utils.set_random_seed(cfg.seed)
    if cfg.train.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    tr_paths, tr_labels = load_manifest(cfg.data.train_manifest, cfg.data)
    va_paths, va_labels = load_manifest(cfg.data.val_manifest, cfg.data)
    _check_files(tr_paths, "train")
    _check_files(va_paths, "val")
    t = cfg.train
    aug = build_augmentation(cfg.augment, cfg.seed) if cfg.augment.enabled else None
    train_ds = make_dataset(
        tr_paths,
        tr_labels,
        cfg.data,
        t.batch_size,
        cfg.model.num_classes,
        shuffle=True,
        repeat=t.steps_per_epoch is not None,
        seed=cfg.seed,
        augment=aug,
        shuffle_buffer=t.shuffle_buffer,
    )
    # With validation_steps set, validate on a shuffled, repeating stream like a shuffled
    # flow_from_directory generator; otherwise on the whole validation set each epoch.
    partial_val = t.validation_steps is not None
    val_ds = make_dataset(
        va_paths,
        va_labels,
        cfg.data,
        t.batch_size,
        cfg.model.num_classes,
        shuffle=partial_val,
        repeat=partial_val,
        seed=cfg.seed + 1,
    )

    model = build_model(cfg.model, cfg.data.image_size)
    compile_model(model, cfg)
    log.info(
        "model %s: %d params (%d trainable)",
        model.name,
        model.count_params(),
        sum(int(np.prod(w.shape)) for w in model.trainable_weights),
    )
    t0 = time.perf_counter()
    hist = model.fit(
        train_ds,
        epochs=t.epochs,
        steps_per_epoch=t.steps_per_epoch,
        validation_data=val_ds,
        validation_steps=t.validation_steps,
        callbacks=[EpochTimer(), keras.callbacks.CSVLogger(str(run_dir / "history.csv"))],
        verbose=2,
    )
    train_seconds = time.perf_counter() - t0
    model.save(run_dir / "model.keras")

    last = {k: float(v[-1]) for k, v in hist.history.items()}
    metrics: dict[str, Any] = {
        "name": cfg.name,
        "seed": cfg.seed,
        "params": int(model.count_params()),
        "trainable_params": int(sum(int(np.prod(w.shape)) for w in model.trainable_weights)),
        "train_seconds": train_seconds,
        "images_seen": int(
            t.epochs * (t.steps_per_epoch or np.ceil(len(tr_paths) / t.batch_size)) * t.batch_size
        ),
        "last_epoch": last,
        "val_full": evaluate_split(model, cfg, cfg.data.val_manifest, "val", run_dir),
    }
    if cfg.data.test_manifest:
        metrics["test_full"] = evaluate_split(model, cfg, cfg.data.test_manifest, "test", run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("done: %s", json.dumps({k: metrics[k] for k in ("last_epoch",)}))
    return metrics


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
