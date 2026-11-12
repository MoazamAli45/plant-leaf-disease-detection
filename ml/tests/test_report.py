"""Results collection and the paper-vs-reproduced table (synthetic run files)."""

import json

import pandas as pd

from leafdx.train.report import PAPER_TABLE3, collect, comparison_table, load_runs, reproduction


def fake_run(root, name, acc):
    d = root / name / "seed42"
    d.mkdir(parents=True)
    last = {"accuracy": acc, "loss": 0.1, "val_accuracy": acc - 0.01, "val_loss": 0.2}
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "name": name,
                "seed": 42,
                "params": 10,
                "trainable_params": 5,
                "images_seen": 100,
                "train_seconds": 60.0,
                "last_epoch": last,
                "val_full": {"accuracy": acc - 0.02, "macro_f1": acc - 0.03},
            }
        )
    )
    pd.DataFrame(
        {"epoch": [0, 1], "accuracy": [0.5, acc], "val_accuracy": [0.4, acc - 0.01]}
    ).to_csv(d / "history.csv", index=False)
    (d / "env.json").write_text(json.dumps({"gpus": []}))
    (d / "config.yaml").write_text("name: x\n")
    (d / "model.keras").write_bytes(b"weights")
    return d


def test_collect_skips_weights(tmp_path):
    fake_run(tmp_path / "outputs", "paper_cnn_steps100", 0.9)
    (tmp_path / "outputs" / "unfinished" / "seed1").mkdir(parents=True)
    copied = collect(tmp_path / "outputs", tmp_path / "results")
    assert len(copied) == 1
    names = {p.name for p in copied[0].iterdir()}
    assert "metrics.json" in names and "model.keras" not in names


def test_comparison_differences(tmp_path):
    fake_run(tmp_path, "paper_mobilenet_fullepochs", 0.96)
    table = comparison_table(load_runs(tmp_path))
    row = table[table["metric"].str.startswith("train accuracy")].iloc[0]
    assert row["paper"] == PAPER_TABLE3["paper_mobilenet"]["accuracy"]
    assert abs(row["difference"] - (0.96 - 0.9348)) < 1e-9


def test_reproduction_outputs(tmp_path):
    fake_run(tmp_path / "results", "paper_cnn_steps100", 0.9)
    fake_run(tmp_path / "results", "paper_mobilenet_steps100", 0.95)
    files = reproduction(tmp_path / "results", tmp_path / "fig")
    assert all(f.exists() for f in files)
    md = (tmp_path / "fig" / "paper_vs_reproduced.md").read_text()
    assert "| cnn | steps100 | 42 | train accuracy" in md
