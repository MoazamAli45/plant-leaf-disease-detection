"""Inference models: the trained network without augmentation layers, in float32.

A Phase 3 ``model.keras`` contains its augmentation stem (inactive at inference) and was saved
under the mixed-float16 policy. For export, Grad-CAM and benchmarking we rebuild the same
architecture from the run's ``config.yaml`` with only the preprocessing layer as stem, under
the float32 policy, and copy the trained weights (augmentation layers have none, so the
weight lists line up). The result maps raw [0, 255] RGB pixels to class probabilities.
"""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np

import leafdx.train.data  # noqa: F401  (registers custom layers for loading)
from leafdx.train.config import ExperimentCfg, load_config
from leafdx.train.data import preprocessing_layer
from leafdx.train.models import build_model


def build_inference_model(cfg: ExperimentCfg) -> keras.Model:
    """Same architecture as training. ``model.weights`` is kept as trained: Keras' EfficientNet
    adds an ImageNet rescaling layer only when built with ``weights="imagenet"``, so building
    with ``weights=None`` would silently change the graph. Weights are then overwritten."""
    stem = [preprocessing_layer(cfg.data.preprocessing)] if cfg.data.augment_in_model else None
    return build_model(cfg.model, cfg.data.image_size, stem=stem)


def check_same_outputs(a: keras.Model, b: keras.Model, size: int, atol: float = 1e-3) -> float:
    """Max |difference| of the two models' outputs on random images; raises if above atol."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (4, size, size, 3)).astype(np.float32)
    diff = float(np.abs(np.asarray(a(x, training=False)) - np.asarray(b(x, training=False))).max())
    if diff > atol:
        raise ValueError(f"rebuilt inference model differs from the trained one (max {diff:.4f})")
    return diff


def load_inference_model(run_dir: Path) -> tuple[keras.Model, ExperimentCfg]:
    """Load ``run_dir/model.keras`` into a clean float32 inference graph."""
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("float32")
    try:
        cfg = load_config(run_dir / "config.yaml")
        trained = keras.models.load_model(run_dir / "model.keras", compile=False)
        model = build_inference_model(cfg)
        src, dst = trained.get_weights(), model.get_weights()
        if [w.shape for w in src] != [w.shape for w in dst]:
            raise ValueError(f"{run_dir}: trained and rebuilt models do not match")
        model.set_weights([np.asarray(w, dtype=d.dtype) for w, d in zip(src, dst, strict=True)])
        # mixed float16 vs float32 compute: differences of ~1e-3 in probabilities are expected
        check_same_outputs(trained, model, cfg.data.image_size, atol=5e-2)
        return model, cfg
    finally:
        keras.mixed_precision.set_global_policy(previous)
