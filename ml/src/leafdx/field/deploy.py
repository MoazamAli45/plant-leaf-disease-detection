"""Export a Phase 6 fine-tuned model to the web app and the API.

    python -m leafdx.field.deploy --run results/p6_ft_p3_mobilenet_v1_kall_replay50/seed0 --tta

1. loads ``model.keras`` from the run and checks it reproduces the run's saved predictions;
2. evaluates it on the full leak-free lab test set (11,599 images) and on field-test, so the
   app's headline numbers use the same lab set as Phase 3; with ``--tta`` the headline numbers
   use test-time augmentation (4 views, as the app does), and single-view numbers are kept too;
3. exports a SavedModel and a TF.js graph model with float16 weights (as in Phase 4), copies
   it to ``web/public/model`` and the Keras file to ``api/model``;
4. writes ``docs/figures/phase6/deployed.json`` with every number the app shows.

Afterwards re-run the browser checks (docs/field.md, "Deployment").
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from leafdx.field import data as fd
from leafdx.field.methods import predict
from leafdx.field.run import evaluate

ROOT = Path(__file__).resolve().parents[4]
TFJS_CONVERTER = Path(".venv-tfjs/bin/tensorflowjs_converter")


def lab_test_leak_free(data_root: Path, cache: Path) -> fd.ImageSet:
    df = pd.read_csv(fd.CLEAN / "test.csv")
    links = pd.read_csv(fd.CLEAN / "test_links_corrected.csv")
    df = df.merge(links[["path", "linked_any"]], on="path")
    return fd.load("lab_test_leak_free", df[~df["linked_any"]], data_root, cache)


def previous_model(base: str, results: Path = Path("results")) -> dict:
    """The Phase 3 model the fine-tuning started from, on the same two test sets: the leak-free
    lab test from its saved Phase 3 predictions, field-test from the Phase 6 baseline run."""
    d = np.load(results / base / "seed42" / "probs_test.npz", allow_pickle=True)
    links = pd.read_csv(fd.CLEAN / "test_links_corrected.csv").set_index("path")["linked_any"]
    keep = ~links.loc[list(d["paths"])].to_numpy()
    base_run = json.loads((results / f"p6_base_{base}" / "seed0" / "metrics.json").read_text())
    return {
        "model": f"{base}, seed 42 (Phase 3)",
        "lab_test_leak_free_accuracy": float(
            (d["probs"][keep].argmax(1) == d["labels"][keep]).mean()
        ),
        "field_test_accuracy": base_run["metrics"]["field_test"]["accuracy"],
        "field_test_ece": base_run["metrics"]["field_test"]["ece"],
    }


def graph_info(tfjs_dir: Path, label: str, tta: bool) -> dict:
    """Node names the app needs for Grad-CAM, found in the converted graph: the classifier's
    dense layer, the pooled feature maps feeding it, and the dense kernel. Written next to the
    model as ``leafdx.json``, so the app works with any backbone that ends in global average
    pooling and one dense softmax layer (MobileNetV1, EfficientNetB0)."""
    nodes = {
        n["name"]: n
        for n in json.loads((tfjs_dir / "model.json").read_text())["modelTopology"]["node"]
    }
    dense = [n for n in nodes.values() if n["op"] == "_FusedMatMul" and "classifier" in n["name"]]
    if len(dense) != 1:
        raise SystemExit(f"expected one classifier dense layer, found {len(dense)}")
    pooled, kernel = dense[0]["input"][0], dense[0]["input"][1]
    if nodes[pooled]["op"] != "Mean":
        raise SystemExit("the classifier is not fed by global average pooling")
    info = {
        "label": label,
        "feature_node": nodes[pooled]["input"][0],
        "kernel_weight": kernel,
        "output_node": "Identity",
        "tta_views": 4 if tta else 1,
    }
    (tfjs_dir / "leafdx.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def update_service_worker(web: Path) -> str:
    """List the model files in the service worker's precache and derive its cache version
    from their content, so browsers drop a cached old model when the model changes."""
    import hashlib
    import re

    model_dir = web / "public" / "model"
    files = sorted(f.name for f in model_dir.iterdir() if f.is_file())
    h = hashlib.sha256()
    for f in files:
        h.update((model_dir / f).read_bytes())
    version = f"leafdx-{h.hexdigest()[:12]}"
    sw = web / "public" / "sw.js"
    s = sw.read_text()
    s = re.sub(r'const VERSION = "[^"]*";', f'const VERSION = "{version}";', s)
    s = re.sub(r'(\n  "/model/[^"]*",)+', "".join(f'\n  "/model/{f}",' for f in files), s, count=1)
    sw.write_text(s)
    return version


def dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--cache", type=Path, default=Path("data/cache/p6"))
    p.add_argument("--export", type=Path, default=Path("models/_export/p6_deployed"))
    p.add_argument("--report", type=Path, default=Path("../docs/figures/phase6"))
    p.add_argument("--tta", action="store_true", help="the app averages 4 views (docs/field.md)")
    p.add_argument("--no-copy", action="store_true", help="export only; leave web/ and api/")
    a = p.parse_args(argv)

    info = json.loads((a.run / "metrics.json").read_text())
    model = keras.models.load_model(a.run / "model.keras", compile=False)

    # 1. the saved file reproduces the run
    val = fd.field("val", a.data_root, a.cache)
    saved = np.load(a.run / "probs_field_val.npz")["probs"]
    diff = float(np.abs(predict(model, val) - saved).max())
    if diff > 1e-4:
        raise SystemExit(f"model.keras does not reproduce the run's predictions (max diff {diff})")

    # 2. headline numbers
    lab = lab_test_leak_free(a.data_root, a.cache)
    test = fd.field("test", a.data_root, a.cache)
    lab_m = evaluate(lab.y, predict(model, lab, tta=a.tta))
    test_m = evaluate(test.y, predict(model, test, tta=a.tta))
    single = None
    if a.tta:
        lab_1 = evaluate(lab.y, predict(model, lab))
        test_1 = evaluate(test.y, predict(model, test))
        single = {
            "lab_test_leak_free_accuracy": lab_1["accuracy"],
            "field_test_accuracy": test_1["accuracy"],
            "field_test_macro_f1": test_1["macro_f1"],
            "field_test_ece": test_1["ece"],
        }

    # 3. exports
    a.export.mkdir(parents=True, exist_ok=True)
    sm = a.export / "saved_model"
    if sm.exists():
        shutil.rmtree(sm)
    model.export(str(sm), verbose=False)
    tfjs = a.export / "tfjs_fp16"
    if tfjs.exists():
        shutil.rmtree(tfjs)
    cmd = [
        str(TFJS_CONVERTER),
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        "--signature_name=serving_default",
        "--quantize_float16=*",
        str(sm),
        str(tfjs),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    shutil.copy(a.run / "model.keras", a.export / "inference.keras")

    label = {"p3_mobilenet_v1": "MobileNetV1", "p3_efficientnet_b0": "EfficientNetB0"}.get(
        info["model"], info["model"]
    )
    graph_info(tfjs, label, a.tta)
    if not a.no_copy:
        web_model = ROOT / "web" / "public" / "model"
        shutil.rmtree(web_model, ignore_errors=True)
        shutil.copytree(tfjs, web_model)
        print("service worker cache version:", update_service_worker(ROOT / "web"))
        api_model = ROOT / "api" / "model"
        api_model.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.run / "model.keras", api_model / "inference.keras")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": str(a.run),
        "base_model": info["model"],
        "method": "fine-tuned on labelled PlantDoc field-train images, with lab replay"
        + (", test-time augmentation (4 views)" if a.tta else ""),
        "tta": a.tta,
        "k_per_class": info.get("k") or "all",
        "seed": info["seed"],
        "n_field_train": info["n_field_train"],
        "n_lab_replay": info["n_lab_replay"],
        "lr": info["lr"],
        "best_epoch": info["best_epoch"],
        "lab_test_leak_free": {
            "n": lab_m["n"],
            "accuracy": lab_m["accuracy"],
            "macro_f1": lab_m["macro_f1"],
            "ece": lab_m["ece"],
        },
        "field_test": {
            "n": test_m["n"],
            "accuracy": test_m["accuracy"],
            "macro_f1": test_m["macro_f1"],
            "crop_accuracy": test_m["crop_accuracy"],
            "ece": test_m["ece"],
        },
        "single_view": single,
        "before_fine_tuning": previous_model(info["model"]),
        "tfjs_fp16_mb": round(dir_mb(tfjs), 2),
        "check_max_prob_diff_vs_run": diff,
    }
    a.report.mkdir(parents=True, exist_ok=True)
    (a.report / "deployed.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
