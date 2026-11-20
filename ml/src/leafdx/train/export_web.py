"""Data for the web app, generated from the ML outputs (never typed by hand).

    python -m leafdx.train.export_web research   # -> web/src/data/research.json
    python -m leafdx.train.export_web verify     # -> web/test-data/ (tensors + Keras outputs)
    python -m leafdx.train.export_web classes    # -> web/src/data/classes.json
    python -m leafdx.train.export_web api-model  # -> api/model/ (inference.keras, classes.json)

``verify`` writes a few clean-test and PlantDoc images, preprocessed exactly as in training,
with the Keras model's probabilities, so ``web/scripts/verify-model.mjs`` can check that the
TF.js model in the browser gives the same answers.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leafdx.data.classes import PLANTVILLAGE_CLASSES, display_name, is_healthy, split_class

WEB = Path("../web")
FIG = Path("../docs/figures")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def classes(out: Path) -> Path:
    data = [
        {
            "index": i,
            "class": c,
            "crop": split_class(c)[0],
            "name": display_name(c),
            "healthy": is_healthy(c),
        }
        for i, c in enumerate(PLANTVILLAGE_CLASSES)
    ]
    out.write_text(json.dumps(data, indent=1) + "\n")
    return out


def api_model(export_dir: Path, out_dir: Path) -> Path:
    """Copy the verified float32 inference model and the class list for the API."""
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(export_dir / "inference.keras", out_dir / "inference.keras")
    classes(out_dir / "classes.json")
    return out_dir


def research(out: Path) -> Path:
    """Every number the research page shows, with the file it came from."""
    audit = json.loads((FIG / "data_audit" / "summary.json").read_text())
    recheck = json.loads((FIG / "data_audit" / "recheck_newpixel.json").read_text())
    explain = json.loads((FIG / "phase3" / "explain" / "summary.json").read_text())
    data: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "audit": "docs/figures/data_audit/summary.json, recheck_newpixel.json",
            "reproduction": "docs/figures/reproduction/paper_vs_reproduced.csv",
            "main_results": "docs/figures/phase3/main_results.csv",
            "field_analysis": "docs/figures/phase3/field_analysis.csv",
            "explain": "docs/figures/phase3/explain/summary.json",
            "efficiency": "docs/figures/phase3/efficiency.csv",
            "selective": "docs/figures/phase3/selective_prediction.csv",
        },
        "audit": {
            "npd_images": audit["npd"]["n_images"],
            "paper_val_images": recheck["paper_split_val"]["n"],
            "paper_val_linked": recheck["paper_split_val"]["n_linked_any"],
            "paper_val_linked_frac": recheck["paper_split_val"]["frac_linked_any"],
            "paper_val_source_frac": recheck["paper_split_val"]["frac_linked_source"],
            "paper_val_leaf_frac": recheck["paper_split_val"]["frac_linked_leaf"],
            "clean_split_counts": audit["npd"]["clean_split_counts"],
            "clean_test_linked_frac_v1": recheck["clean_split_v1_test"]["frac_linked_any"],
            "plantdoc_test_images": audit["plantdoc"]["n_test"],
        },
        "reproduction": _csv(FIG / "reproduction" / "paper_vs_reproduced.csv"),
        "main_results": _csv(FIG / "phase3" / "main_results.csv"),
        "field_analysis": _csv(FIG / "phase3" / "field_analysis.csv"),
        "explain": explain,
        "efficiency": _csv(FIG / "phase3" / "efficiency.csv"),
        "selective": _csv(FIG / "phase3" / "selective_prediction.csv"),
    }
    data["tfjs_verify"] = {
        f.stem.removeprefix("tfjs_verify_mnv1_"): json.loads(f.read_text())
        for f in sorted((FIG / "phase3").glob("tfjs_verify_mnv1_*.json"))
    }
    data["sources"]["tfjs_verify"] = "docs/figures/phase3/tfjs_verify_mnv1_*.json"
    abl = FIG / "phase3" / "ablations.csv"
    data["ablations"] = _csv(abl) if abl.exists() else None
    # Phase 6 (field adaptation): summary table, field split and, once chosen, the deployed model
    p6 = FIG / "phase6"
    data["field_adaptation"] = (
        {
            "results": _csv(p6 / "results.csv"),
            "split": json.loads((p6 / "field_split.json").read_text()),
        }
        if (p6 / "results.csv").exists()
        else None
    )
    deployed = p6 / "deployed.json"
    data["deployed"] = json.loads(deployed.read_text()) if deployed.exists() else None
    data["sources"]["field_adaptation"] = "docs/figures/phase6/results.csv, field_split.json"
    data["sources"]["deployed"] = "docs/figures/phase6/deployed.json"
    out.write_text(json.dumps(data, indent=1) + "\n")
    return out


def _reference_cams(model: object, x: np.ndarray, class_index: np.ndarray) -> np.ndarray:
    """Grad-CAM at the feature-map resolution (7 x 7 for MobileNetV1), scaled to a maximum of 1,
    computed with TensorFlow's automatic gradient. The web app computes the same map without
    gradients (web/src/lib/gradcam.ts); scripts/verify-model.mjs compares the two."""
    import tensorflow as tf

    from leafdx.explain.gradcam import gradcam_model

    grad_model = gradcam_model(model)
    out = []
    for xi, k in zip(x, class_index, strict=True):
        with tf.GradientTape() as tape:
            feats, p = grad_model(tf.convert_to_tensor(xi[None]), training=False)
            score = p[:, int(k)]
        grads = tf.cast(tape.gradient(score, feats), tf.float32)
        w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(w * tf.cast(feats, tf.float32), axis=-1))[0].numpy()
        out.append(cam / cam.max() if cam.max() > 0 else cam)
    return np.stack(out)


def verify(model_dir: Path, out_dir: Path, per_class: int, n_field: int, seed: int) -> Path:
    import tensorflow as tf

    from leafdx.train.inference import load_inference_model

    if model_dir.suffix == ".keras":  # a plain Keras file, e.g. a Phase 6 fine-tuned model
        import keras

        model = keras.models.load_model(model_dir, compile=False)
    else:
        model, _ = load_inference_model(model_dir)
    test = pd.read_csv("manifests/clean_split/test.csv")
    links = pd.read_csv("manifests/clean_split/test_links_corrected.csv")
    test = test.merge(links[["path", "linked_any"]], on="path")
    test = test[~test["linked_any"]].sample(frac=1.0, random_state=seed)
    test = test.groupby("label").head(per_class)
    # field images from the Phase 6 field-test split when it exists, so a model fine-tuned on
    # field-train images is never checked on images it was trained on
    field_csv = Path("manifests/field_split/test.csv")
    if not field_csv.exists():
        field_csv = Path("manifests/plantdoc/test.csv")
    field = pd.read_csv(field_csv).sample(n_field, random_state=seed)
    rows = pd.concat([test.assign(set="test"), field.assign(set="plantdoc")])

    def load(p: str) -> np.ndarray:
        img = tf.io.decode_image(
            tf.io.read_file(str(Path("data") / p)), channels=3, expand_animations=False
        )
        return tf.image.resize(img, (224, 224), method="bilinear").numpy()

    x = np.stack([load(p) for p in rows["path"]]).astype(np.float32)
    probs = model.predict(x, verbose=0).astype(np.float32)
    # test-time augmentation as in Phase 6 (the app averages the same 4 views, web/src/lib/tta.ts)
    from leafdx.field.data import ImageSet
    from leafdx.field.methods import predict as predict_views

    probs_tta = predict_views(model, ImageSet("verify", x, np.zeros(len(x), int), []), tta=True)
    cams = _reference_cams(model, x, probs.argmax(axis=1))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gradcam.json").write_text(
        json.dumps({"class_index": probs.argmax(axis=1).tolist(), "cams": cams.round(5).tolist()})
    )
    (out_dir / "images.f32").write_bytes(x.tobytes())
    (out_dir / "expected.json").write_text(
        json.dumps(
            {
                "shape": list(x.shape),
                "model": str(model_dir),
                "labels": [PLANTVILLAGE_CLASSES.index(lab) for lab in rows["label"]],
                "sets": rows["set"].tolist(),
                "probs": probs.round(6).tolist(),
                "probs_tta": probs_tta.round(6).tolist(),
            }
        )
    )
    return out_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("research")
    sub.add_parser("classes")
    am = sub.add_parser("api-model")
    am.add_argument("--export", type=Path, default=Path("models/_export/p3_mobilenet_v1"))
    v = sub.add_parser("verify")
    v.add_argument(
        "--model",
        type=Path,
        default=Path("models/p3_mobilenet_v1"),
        help="a Phase 3 run directory, or a .keras file",
    )
    v.add_argument("--per-class", type=int, default=2)
    v.add_argument("--field", type=int, default=24)
    v.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.cmd == "research":
        print("wrote", research(WEB / "src" / "data" / "research.json"))
    elif args.cmd == "api-model":
        print("wrote", api_model(args.export, Path("../api/model")))
    elif args.cmd == "classes":
        print("wrote", classes(WEB / "src" / "data" / "classes.json"))
    else:
        print("wrote", verify(args.model, WEB / "test-data", args.per_class, args.field, args.seed))


if __name__ == "__main__":
    main()
