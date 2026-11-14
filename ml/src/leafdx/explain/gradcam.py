"""Grad-CAM (Selvaraju et al. 2017) and quantitative leaf-vs-background checks.

    python -m leafdx.explain.gradcam --models models --per-class 5

For a stratified sample of clean-test originals, each model gets:

- **Grad-CAM leaf ratio**: share of Grad-CAM mass on the leaf / share of the image that is leaf.
  1 = no preference; above 1 = attends to the leaf more than its area would predict.
- **Counterfactual accuracies** (chance = 1/38):
  - *background only*: the leaf (mask dilated by 6 px, to remove its outline and halo) is
    filled with the median background colour. The silhouette may still hint at leaf shape;
  - *border only*: only a 16-px frame of the original photo is kept and the interior is filled
    with the frame's median colour, so no leaf shape remains (close to Noyan 2022);
  - *leaf only*: the background is replaced by uniform grey.

**Leaf masks** are the PlantVillage authors' own segmentations (``raw/segmented`` in the
upstream repository, Mohanty et al. 2016), fetched for the sampled images only and cached in
``data/raw/plantvillage/segmented``. Lesions are not annotated in PlantVillage, so whether a
model looks at lesions (rather than the leaf in general) is only shown qualitatively.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from PIL import Image
from scipy import ndimage

from leafdx.data.classes import CLASS_TO_INDEX, PLANTVILLAGE_CLASSES, display_name, is_healthy
from leafdx.data.sources import PLANTVILLAGE_GIT_SHA
from leafdx.train.inference import load_inference_model

SIZE = 224
BORDER = 16
LABELS = {
    "p3_cnn_softmax": "Paper CNN (softmax)",
    "p3_mobilenet_v1": "MobileNetV1",
    "p3_mobilenet_v2": "MobileNetV2",
    "p3_mobilenet_v3_small": "MobileNetV3-Small",
    "p3_efficientnet_b0": "EfficientNetB0",
}
SEGMENTED_URL = (
    "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/"
    f"{PLANTVILLAGE_GIT_SHA}/raw/segmented/"
)


# ----------------------------------------------------------------------------- masks


def segmented_path(data_root: Path, rel_path: str, label: str) -> Path:
    stem = Path(rel_path).stem
    return data_root / "raw" / "plantvillage" / "segmented" / label / f"{stem}_final_masked.jpg"


def fetch_segmented(data_root: Path, rel_path: str, label: str) -> Path | None:
    """Download the upstream segmented image for one photo (cached). None if unavailable."""
    dest = segmented_path(data_root, rel_path, label)
    if dest.exists():
        return dest
    url = SEGMENTED_URL + urllib.parse.quote(f"{label}/{dest.name}")
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def leaf_mask(seg_path: Path) -> np.ndarray:
    """Leaf mask at SIZE x SIZE from a segmented image (background is black)."""
    with Image.open(seg_path) as im:
        seg = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.NEAREST))
    m = seg.astype(int).sum(axis=-1) > 30
    m = ndimage.binary_opening(m, iterations=1)
    return ndimage.binary_fill_holes(m)


# ----------------------------------------------------------------------------- Grad-CAM


def target_layer(model: keras.Model) -> str:
    """The last convolutional block: the backbone (transfer models) or conv4 (paper CNN)."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model | keras.layers.Conv2D):
            return layer.name
    raise ValueError("no convolutional layer found")


def gradcam_model(model: keras.Model) -> keras.Model:
    """(features, probabilities) model. Our models are linear chains, so the graph is rebuilt by
    calling the layers in order; this also works when the target is a nested backbone, whose
    ``.output`` would refer to its own inner graph rather than its use in ``model``."""
    target = target_layer(model)
    inp = keras.Input(model.inputs[0].shape[1:])
    h, feats = inp, None
    for layer in model.layers[1:]:
        h = layer(h)
        if layer.name == target:
            feats = h
    return keras.Model(inp, [feats, h])


def gradcam(
    grad_model: keras.Model, x: np.ndarray, class_index: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Heat map in [0, 1] at input resolution, and the class probabilities. ``x`` is a single
    image in [0, 255], shape (H, W, 3); the model contains its own preprocessing."""
    inp = tf.convert_to_tensor(x[None].astype(np.float32))
    with tf.GradientTape() as tape:
        feats, probs = grad_model(inp, training=False)
        k = int(tf.argmax(probs[0])) if class_index is None else class_index
        score = probs[:, k]
    grads = tape.gradient(score, feats)
    if grads is None:
        raise RuntimeError("no gradient to the target layer")
    feats = tf.cast(feats, tf.float32)
    grads = tf.cast(grads, tf.float32)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * feats, axis=-1))[0].numpy()
    cam = np.asarray(Image.fromarray(cam).resize(x.shape[1::-1], Image.Resampling.BILINEAR))
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam, probs[0].numpy()


# ----------------------------------------------------------------------------- counterfactuals


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR))


def background_only(rgb: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    grown = ndimage.binary_dilation(leaf, iterations=6)
    out = rgb.copy()
    fill = np.median(rgb[~grown], axis=0) if (~grown).any() else np.full(3, 128.0)
    out[grown] = fill.astype(np.uint8)
    return out


def border_only(rgb: np.ndarray, width: int = BORDER) -> np.ndarray:
    frame = np.ones(rgb.shape[:2], dtype=bool)
    frame[width:-width, width:-width] = False
    out = rgb.copy()
    out[~frame] = np.median(rgb[frame], axis=0).astype(np.uint8)
    return out


def leaf_only(rgb: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[~leaf] = 128
    return out


# ----------------------------------------------------------------------------- study


def sample_test(manifest: Path, per_class: int, seed: int) -> pd.DataFrame:
    """Stratified sample of test originals (excluding split v1's mislabelled newPixel copies)."""
    df = pd.read_csv(manifest, keep_default_na=False)
    df = df[(df["aug_type"] == "original") & ~df["path"].str.contains("_newPixel")]
    return df.groupby("label", group_keys=False).sample(per_class, random_state=seed)


def attach_masks(sample: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    segs = [fetch_segmented(data_root, r.path, r.label) for r in sample.itertuples(index=False)]
    out = sample.assign(segmented=segs)
    missing = out["segmented"].isna().sum()
    if missing:
        print(f"no upstream segmentation for {missing} sampled images; they are skipped")
    return out[out["segmented"].notna()]


def study(model_dir: Path, sample: pd.DataFrame, data_root: Path) -> tuple[dict, pd.DataFrame]:
    model, _ = load_inference_model(model_dir)
    gm = gradcam_model(model)
    rows = []
    for r in sample.itertuples(index=False):
        rgb = load_image(data_root / r.path)
        leaf = leaf_mask(Path(r.segmented))
        cam, probs = gradcam(gm, rgb)
        y = CLASS_TO_INDEX[r.label]
        batch = np.stack([background_only(rgb, leaf), border_only(rgb), leaf_only(rgb, leaf)])
        cf = model.predict(batch.astype(np.float32), verbose=0).argmax(axis=1)
        area = float(leaf.mean())
        energy = float((cam * leaf).sum() / max(cam.sum(), 1e-9))
        rows.append(
            {
                "path": r.path,
                "label": r.label,
                "y": y,
                "pred": int(probs.argmax()),
                "leaf_area": area,
                "leaf_energy": energy,
                "leaf_energy_ratio": energy / area if area > 0 else np.nan,
                "background_only_correct": bool(cf[0] == y),
                "border_only_correct": bool(cf[1] == y),
                "leaf_only_correct": bool(cf[2] == y),
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "n": len(df),
        "accuracy": float((df["pred"] == df["y"]).mean()),
        "leaf_area_mean": float(df["leaf_area"].mean()),
        "leaf_energy_mean": float(df["leaf_energy"].mean()),
        "leaf_energy_ratio_median": float(df["leaf_energy_ratio"].median()),
        "background_only_accuracy": float(df["background_only_correct"].mean()),
        "border_only_accuracy": float(df["border_only_correct"].mean()),
        "leaf_only_accuracy": float(df["leaf_only_correct"].mean()),
        "chance": 1 / len(PLANTVILLAGE_CLASSES),
    }
    return summary, df


# ----------------------------------------------------------------------------- figures


def fig_masks(sample: pd.DataFrame, data_root: Path, out: Path, n: int = 8) -> Path:
    picks = sample.groupby("label").head(1).head(n)
    fig, axes = plt.subplots(4, n, figsize=(1.6 * n, 7))
    for j, r in enumerate(picks.itertuples(index=False)):
        rgb = load_image(data_root / r.path)
        leaf = leaf_mask(Path(r.segmented))
        dim = rgb.copy()
        dim[~leaf] = (dim[~leaf] * 0.25).astype(np.uint8)
        for i, img in enumerate([rgb, dim, background_only(rgb, leaf), border_only(rgb)]):
            axes[i, j].imshow(img)
        axes[0, j].set_title(display_name(r.label)[:22], fontsize=6)
    for ax, lab in zip(
        axes[:, 0], ["original", "leaf mask", "background only", "border only"], strict=True
    ):
        ax.set_ylabel(lab, fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Upstream leaf masks and the counterfactual inputs",
        x=0.01,
        ha="left",
        fontsize=10,
        weight="bold",
    )
    p = out / "masks.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_examples(
    model_dirs: dict[str, Path], paths: list[str], data_root: Path, out: Path, name: str, title: str
) -> Path:
    models = {k: gradcam_model(load_inference_model(d)[0]) for k, d in model_dirs.items()}
    fig, axes = plt.subplots(
        len(paths),
        len(models) + 1,
        figsize=(1.9 * (len(models) + 1), 1.95 * len(paths)),
        squeeze=False,
    )
    for i, rel in enumerate(paths):
        rgb = load_image(data_root / rel)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_ylabel(Path(rel).parent.name[:24], fontsize=6)
        for j, (key, gm) in enumerate(models.items(), start=1):
            cam, probs = gradcam(gm, rgb)
            axes[i, j].imshow(rgb)
            axes[i, j].imshow(cam, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
            pred = display_name(PLANTVILLAGE_CLASSES[int(probs.argmax())])
            axes[i, j].set_title(f"{pred[:24]}\n{probs.max():.0%}", fontsize=5.5)
            if i == 0:
                axes[i, j].text(
                    0.5,
                    1.32,
                    LABELS.get(key, key),
                    transform=axes[i, j].transAxes,
                    ha="center",
                    fontsize=7.5,
                    weight="bold",
                )
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, weight="bold", y=1.02)
    p = out / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--test-manifest", type=Path, default=Path("manifests/clean_split/test.csv"))
    p.add_argument("--plantdoc-manifest", type=Path, default=Path("manifests/plantdoc/test.csv"))
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("../docs/figures/phase3/explain"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    sample = attach_masks(
        sample_test(args.test_manifest, args.per_class, args.seed), args.data_root
    )
    sample.drop(columns="segmented").to_csv(args.out / "sample.csv", index=False)
    print("wrote", fig_masks(sample, args.data_root, args.out))
    dirs = {k: args.models / k for k in LABELS if (args.models / k / "model.keras").exists()}
    summaries = {}
    for key, d in dirs.items():
        s, df = study(d, sample, args.data_root)
        df.to_csv(args.out / f"per_image_{key}.csv", index=False)
        summaries[key] = s
        print(key, json.dumps({k: round(v, 3) for k, v in s.items()}))
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))
    pd.DataFrame([{"model": LABELS[k], **v} for k, v in summaries.items()]).to_csv(
        args.out / "summary.csv", index=False
    )

    diseased = sample[~sample["label"].map(is_healthy)].groupby("label").head(1)
    lab = diseased.sample(6, random_state=args.seed)["path"].tolist()
    print(
        "wrote",
        fig_examples(
            dirs,
            lab,
            args.data_root,
            args.out,
            "gradcam_lab",
            "Grad-CAM on clean-test images (diseased classes, seed-42 models)",
        ),
    )
    pdm = pd.read_csv(args.plantdoc_manifest, keep_default_na=False)
    field = pdm[pdm["match"] == "exact"].sample(6, random_state=args.seed)["path"].tolist()
    print(
        "wrote",
        fig_examples(
            dirs,
            field,
            args.data_root,
            args.out,
            "gradcam_field",
            "Grad-CAM on PlantDoc field images (seed-42 models)",
        ),
    )


if __name__ == "__main__":
    main()
