"""Model size, CPU latency and accuracy before/after conversion (TFLite, INT8, TF.js).

    python -m leafdx.train.efficiency --models models

For each ``models/<run>/`` (seed-42 weights):
1. rebuild the float32 inference model (``leafdx.train.inference``), export a SavedModel;
2. convert to TFLite: float32, dynamic-range (int8 weights) and full INT8 (int8 weights and
   activations, float input/output; calibrated on 200 training images);
3. convert to TF.js graph models (float32 and uint8-quantised weights) with the separate
   ``.venv-tfjs`` environment (tensorflowjs pins an older TensorFlow);
4. measure CPU latency (batch 1, 4 threads, median and p90 of 50 runs after 10 warm-up runs)
   and accuracy on a stratified sample of the leak-free clean test set.

TF.js latency depends on the browser and backend; it is measured in Phase 4.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from leafdx.data.classes import CLASS_TO_INDEX
from leafdx.train.inference import load_inference_model

THREADS = 4
SIZE = 224
TFJS_CONVERTER = Path(".venv-tfjs/bin/tensorflowjs_converter")


def dir_size(p: Path) -> int:
    return (
        sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if p.is_dir()
        else p.stat().st_size
    )


def load_batch(paths: list[str], data_root: Path) -> np.ndarray:
    def one(p: str) -> np.ndarray:
        img = tf.io.decode_image(
            tf.io.read_file(str(data_root / p)), channels=3, expand_animations=False
        )
        return tf.image.resize(img, (SIZE, SIZE), method="bilinear").numpy()

    return np.stack([one(p) for p in paths]).astype(np.float32)


def eval_sample(manifest: Path, links: Path, per_class: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(manifest, keep_default_na=False)
    lk = pd.read_csv(links)
    df = df.merge(lk[["path", "linked_any"]], on="path")
    df = df[~df["linked_any"]]
    # up to per_class per class: a few classes have fewer leak-free test images
    return df.sample(frac=1.0, random_state=seed).groupby("label").head(per_class)


def calib_sample(manifest: Path, n: int, seed: int) -> list[str]:
    return pd.read_csv(manifest).sample(n, random_state=seed)["path"].tolist()


# ----------------------------------------------------------------------------- conversion


def to_tflite(saved_model: Path, out: Path, mode: str, calib: np.ndarray | None = None) -> Path:
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    if mode in ("dynamic", "int8"):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if mode == "int8":
        assert calib is not None

        def rep() -> Iterator[list[np.ndarray]]:
            for i in range(len(calib)):
                yield [calib[i : i + 1]]

        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    out.write_bytes(conv.convert())
    return out


def to_tfjs(saved_model: Path, out: Path, quantize: bool) -> Path | None:
    if not TFJS_CONVERTER.exists():
        return None
    if out.exists():
        shutil.rmtree(out)
    cmd = [
        str(TFJS_CONVERTER),
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        "--signature_name=serving_default",
    ]
    if quantize:
        cmd.append("--quantize_uint8=*")
    r = subprocess.run([*cmd, str(saved_model), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        print("tfjs conversion failed:", r.stderr[-800:])
        return None
    return out


# ----------------------------------------------------------------------------- measurement


def latency_ms(fn: Any, x: np.ndarray, warmup: int = 10, runs: int = 50) -> dict[str, float]:
    for _ in range(warmup):
        fn(x)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) * 1000)
    return {"median": float(np.median(t)), "p90": float(np.percentile(t, 90))}


class TFLiteRunner:
    """TFLite interpreter. Uses the default XNNPACK CPU delegate; if XNNPACK cannot prepare a
    graph (seen for MobileNetV3-Small), falls back to the built-in kernels and records it."""

    def __init__(self, path: Path) -> None:
        self.xnnpack = True
        try:
            self.it = tf.lite.Interpreter(model_path=str(path), num_threads=THREADS)
            self.it.allocate_tensors()
        except RuntimeError as e:
            print(f"XNNPACK failed for {path.name} ({str(e)[:80]}); using built-in kernels")
            self.xnnpack = False
            self.it = tf.lite.Interpreter(
                model_path=str(path),
                num_threads=THREADS,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            )
            self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.it.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        self.it.invoke()
        return self.it.get_tensor(self.out["index"])

    def predict(self, xs: np.ndarray) -> np.ndarray:
        return np.concatenate([self(xs[i : i + 1]) for i in range(len(xs))])


def measure(
    run_dir: Path, sample: pd.DataFrame, calib_paths: list[str], data_root: Path, out_root: Path
) -> list[dict[str, Any]]:
    model, _ = load_inference_model(run_dir)
    name = run_dir.name
    exp = out_root / name
    exp.mkdir(parents=True, exist_ok=True)
    keras_path = exp / "inference.keras"
    model.save(keras_path)
    saved = exp / "saved_model"
    if saved.exists():
        shutil.rmtree(saved)
    model.export(str(saved), verbose=False)

    x_eval = load_batch(sample["path"].tolist(), data_root)
    y = sample["label"].map(CLASS_TO_INDEX).to_numpy()
    calib = load_batch(calib_paths, data_root)
    x1 = x_eval[:1]

    keras_fn = tf.function(lambda t: model(t, training=False))
    ref = np.concatenate([keras_fn(x_eval[i : i + 32]).numpy() for i in range(0, len(x_eval), 32)])
    rows = [
        {
            "model": name,
            "format": "Keras (float32)",
            "params": int(model.count_params()),
            "size_mb": keras_path.stat().st_size / 1e6,
            **_lat(latency_ms(keras_fn, x1)),
            "accuracy": float((ref.argmax(1) == y).mean()),
            "agreement_with_keras": 1.0,
        }
    ]
    for mode, label in [
        ("fp32", "TFLite float32"),
        ("dynamic", "TFLite dynamic-range"),
        ("int8", "TFLite INT8"),
    ]:
        path = to_tflite(saved, exp / f"model_{mode}.tflite", mode, calib)
        r = TFLiteRunner(path)
        pred = r.predict(x_eval)
        rows.append(
            {
                "model": name,
                "format": label,
                "params": int(model.count_params()),
                "size_mb": path.stat().st_size / 1e6,
                **_lat(latency_ms(r, x1)),
                "accuracy": float((pred.argmax(1) == y).mean()),
                "agreement_with_keras": float((pred.argmax(1) == ref.argmax(1)).mean()),
                "xnnpack": r.xnnpack,
            }
        )
    for quant, label in [(False, "TF.js float32"), (True, "TF.js uint8 weights")]:
        p = to_tfjs(saved, exp / ("tfjs_uint8" if quant else "tfjs"), quant)
        if p is not None:
            rows.append(
                {
                    "model": name,
                    "format": label,
                    "params": int(model.count_params()),
                    "size_mb": dir_size(p) / 1e6,
                }
            )
    return rows


def _lat(d: dict[str, float]) -> dict[str, float]:
    return {"cpu_latency_ms_median": d["median"], "cpu_latency_ms_p90": d["p90"]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--export-dir", type=Path, default=Path("models/_export"))
    p.add_argument("--per-class", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("../docs/figures/phase3"))
    args = p.parse_args(argv)
    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    sample = eval_sample(
        Path("manifests/clean_split/test.csv"),
        Path("manifests/clean_split/test_links_corrected.csv"),
        args.per_class,
        args.seed,
    )
    calib = calib_sample(Path("manifests/clean_split/train.csv"), 200, args.seed)
    rows = []
    for d in sorted(args.models.glob("p3_*")):
        if (d / "model.keras").exists():
            print("==", d.name, flush=True)
            rows += measure(d, sample, calib, args.data_root, args.export_dir)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "efficiency.csv", index=False)
    show = df.copy()
    for c in ("size_mb", "cpu_latency_ms_median", "cpu_latency_ms_p90"):
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    for c in ("accuracy", "agreement_with_keras"):
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v * 100:.2f}")
    head = "| " + " | ".join(show.columns) + " |\n|" + "|".join("---" for _ in show.columns) + "|\n"
    body = "".join(
        "| " + " | ".join(str(v) for v in r) + " |\n" for r in show.itertuples(index=False)
    )
    (args.out / "efficiency.md").write_text(head + body)
    meta = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "threads": THREADS,
        "eval_images": len(sample),
        "calibration_images": len(calib),
        "tfjs_converter": TFJS_CONVERTER.exists(),
    }
    (args.out / "efficiency_meta.json").write_text(json.dumps(meta, indent=2))
    print(head + body)


if __name__ == "__main__":
    main()
