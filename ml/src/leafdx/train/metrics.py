"""Classification metrics from predicted probabilities (no TensorFlow needed)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def top_k_accuracy(y: np.ndarray, probs: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def classification_metrics(
    y: np.ndarray, probs: np.ndarray, class_names: Sequence[str]
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Summary dict, per-class table and confusion matrix.

    Accuracy is top-1 over argmax of ``probs``. For sigmoid outputs this is what a user of the
    model sees (the predicted class), independent of which metric Keras reported in training.
    """
    labels = np.arange(len(class_names))
    pred = probs.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame(
        {"class": list(class_names), "precision": p, "recall": r, "f1": f, "support": s}
    )
    summary = {
        "n": len(y),
        "accuracy": float((pred == y).mean()),
        "top3_accuracy": top_k_accuracy(y, probs, 3),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "prob_row_sum_mean": float(probs.sum(axis=1).mean()),  # ~1 for softmax only
    }
    return summary, per_class, confusion_matrix(y, pred, labels=labels)
