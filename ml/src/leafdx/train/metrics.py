"""Classification metrics from predicted probabilities (no TensorFlow needed)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def top_k_accuracy(y: np.ndarray, probs: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """ECE of the top-1 confidence (Guo et al. 2017), equal-width bins."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_metrics(
    y: np.ndarray, probs: np.ndarray, class_names: Sequence[str]
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Summary dict, per-class table and confusion matrix.

    - ``accuracy``: top-1 over all classes (argmax of ``probs``).
    - ``restricted_accuracy``: argmax over the classes that occur in ``y`` only. Relevant
      when the evaluation set covers a subset of the classes (PlantDoc: 28 of 38).
    - Macro averages are over the classes that occur in ``y``.
    - ``ece``: calibration error; meaningful for softmax outputs.
    """
    labels = np.arange(len(class_names))
    pred = probs.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame(
        {"class": list(class_names), "precision": p, "recall": r, "f1": f, "support": s}
    )
    present = s > 0
    restricted = np.where(present[None, :], probs, -np.inf).argmax(axis=1)
    summary = {
        "n": len(y),
        "n_classes_present": int(present.sum()),
        "accuracy": float((pred == y).mean()),
        "restricted_accuracy": float((restricted == y).mean()),
        "top3_accuracy": top_k_accuracy(y, probs, 3),
        "macro_f1": float(f[present].mean()),
        "weighted_f1": float(np.average(f, weights=s)),
        "macro_precision": float(p[present].mean()),
        "macro_recall": float(r[present].mean()),
        "ece": expected_calibration_error(y, probs),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "prob_row_sum_mean": float(probs.sum(axis=1).mean()),  # ~1 for softmax only
    }
    return summary, per_class, confusion_matrix(y, pred, labels=labels)
