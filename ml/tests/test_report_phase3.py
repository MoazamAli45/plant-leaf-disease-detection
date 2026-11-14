"""Phase 3 report on synthetic run directories."""

import json

import numpy as np
import pandas as pd

from leafdx.data.classes import PLANTVILLAGE_CLASSES
from leafdx.train.report_phase3 import build, field_analysis, load, results_table

C = len(PLANTVILLAGE_CLASSES)


def ev(acc):
    sub = {"accuracy": acc}
    return {
        "accuracy": acc,
        "restricted_accuracy": acc,
        "macro_f1": acc,
        "ece": 0.01,
        "subsets": {"original": sub, "exact": sub},
    }


def fake(root, name, seed, test_acc, pd_acc, rng):
    d = root / name / f"seed{seed}"
    d.mkdir(parents=True)
    m = {
        "name": name,
        "seed": seed,
        "params": 1_000_000,
        "epochs_run": 5,
        "best_epoch": 3,
        "eval": {"test": ev(test_acc), "plantdoc": ev(pd_acc)},
    }
    (d / "metrics.json").write_text(json.dumps(m))
    for split, n in (("test", 76), ("plantdoc", 40)):
        y = np.arange(n) % (C if split == "test" else 5)
        probs = rng.dirichlet(np.ones(C), size=n).astype(np.float16)
        np.savez(
            d / f"probs_{split}.npz",
            probs=probs,
            labels=y,
            paths=np.array([f"p{i}" for i in range(n)]),
        )
        pd.DataFrame(
            {
                "class": PLANTVILLAGE_CLASSES,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "support": 2,
            }
        ).to_csv(d / f"per_class_{split}.csv", index=False)
        pd.DataFrame(
            np.eye(C, dtype=int), index=PLANTVILLAGE_CLASSES, columns=PLANTVILLAGE_CLASSES
        ).to_csv(d / f"confusion_{split}.csv")


def test_table_mean_std_and_figures(tmp_path):
    rng = np.random.default_rng(0)
    for s, (t, p) in zip((42, 43, 44), [(0.98, 0.2), (0.96, 0.1), (0.97, 0.15)], strict=True):
        fake(tmp_path / "r", "p3_mobilenet_v1", s, t, p, rng)
    fake(tmp_path / "r", "p3_cnn_softmax", 42, 0.9, 0.05, rng)
    runs = load(tmp_path / "r")
    tbl = results_table(runs, {"p3_mobilenet_v1": "MobileNetV1"})
    assert tbl.loc[0, "test acc. (%)"] == "97.00 ± 1.00"
    assert "leak-free" in " ".join(tbl.columns)
    made = build(tmp_path / "r", tmp_path / "out")
    assert all(p.exists() for p in made)
    assert (tmp_path / "out" / "per_class" / "p3_cnn_softmax_plantdoc.csv").exists()


def test_field_analysis_keys(tmp_path):
    fake(tmp_path, "p3_x", 42, 0.9, 0.1, np.random.default_rng(1))
    fa = field_analysis(load(tmp_path)["p3_x"][0])
    assert 0 <= fa["crop_accuracy"] <= 1
    assert fa["share_predicted_absent_classes"] > 0.5  # random preds over 38 vs 5 present
