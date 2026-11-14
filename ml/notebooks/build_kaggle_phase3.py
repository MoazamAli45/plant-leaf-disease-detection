"""Generate notebooks/kaggle_phase3.ipynb (edit this file, not the notebook).

    python notebooks/build_kaggle_phase3.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("kaggle_phase3.ipynb")

INTRO = """# Phase 3 on Kaggle: corrected and extended experiments

Trains the Phase 3 models on the **clean grouped split**, evaluates them on the clean test set
and on **PlantDoc** field images, and packs the results for download.

**Setup:** *Accelerator* **GPU T4 x2**, *Internet* **on**, *Add Input* **New Plant Diseases
Dataset** (vipoooool).

In the settings cell, choose `BATCH` and `SEEDS`, then **Save Version → Save & Run All
(Commit)**. Kaggle stops a session after 12 hours, so keep each version within the times below
(measured on 2 × T4; two runs go in parallel).

| `BATCH` | runs per seed | `SEEDS` per version | about |
|---|---|---|---|
| `"main"` | 5 models: paper CNN (softmax), MobileNetV1, V2, V3-Small, EfficientNetB0 | one seed | 5–6 h |
| `"mnv3"` | MobileNetV3-Small only | `[42, 43, 44]` | 2.5 h |
| `"ablations"` | 4 MobileNetV1 variants: no augmentation, frozen backbone, width 0.75, width 0.5 | at most two seeds | 3 h per seed |

What happens:
1. **Smoke pass:** every config trains for a few steps (a few minutes), so a bug fails
   early.
2. **Real runs:** two queues, one per GPU. No new run starts after `START_DEADLINE_H` hours,
   and results are packed after every finished run.
3. **Outputs:** `results_<batch>_seed<s>.zip` (metrics, committed to the repo) and
   `models_<batch>_seed<s>.zip` (weights, seed 42 only; used for Grad-CAM and efficiency).

**Only edit the settings cell.** Keep the other cells as they are, in this order."""

SETTINGS = '''# ---- settings ---------------------------------------------------------------------------
REPO_URL = "https://github.com/MoazamAli45/plant-leaf-disease-detection.git"
BRANCH = "main"
BATCH = "mnv3"          # "main", "mnv3" or "ablations"
SEEDS = [42, 43, 44]    # see the table in the first cell for what fits in one version
SMOKE_TEST = True       # quick pass over every config before the real runs
START_DEADLINE_H = 9.0  # no new run starts after this (Kaggle stops sessions at 12 h)

BATCHES = {
    # config -> relative cost, used to balance the two GPU queues
    "main": {
        "configs/phase3/efficientnet_b0.yaml": 3,
        "configs/phase3/mobilenet_v1.yaml": 2,
        "configs/phase3/mobilenet_v2.yaml": 2,
        "configs/phase3/cnn_softmax.yaml": 2,
        "configs/phase3/mobilenet_v3_small.yaml": 1,
    },
    # MobileNetV3-Small alone: the three runs cut off by the first (12 h) version
    "mnv3": {
        "configs/phase3/mobilenet_v3_small.yaml": 1,
    },
    "ablations": {
        "configs/phase3/abl_mnv1_noaug.yaml": 2,
        "configs/phase3/abl_mnv1_frozen.yaml": 1,
        "configs/phase3/abl_mnv1_alpha075.yaml": 2,
        "configs/phase3/abl_mnv1_alpha050.yaml": 1,
    },
}
# weights kept for Grad-CAM and the efficiency study (seed 42 only)
KEEP_MODELS = ["p3_cnn_softmax", "p3_mobilenet_v1", "p3_mobilenet_v2", "p3_mobilenet_v3_small",
               "p3_efficientnet_b0", "p3_abl_mnv1_alpha050", "p3_abl_mnv1_alpha075",
               "p3_abl_mnv1_frozen"]
TAG = f"{BATCH}_seed{'-'.join(map(str, SEEDS))}"
print("this version:", TAG)'''

SETUP = '''# ---- code, NPD and PlantDoc ----------------------------------------------------------------
import glob, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

WORK = Path("/kaggle/working")
repo = WORK / "repo"
if not repo.exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(repo)], check=True)
ML = repo / "ml"
os.chdir(ML)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                        text=True).stdout.strip()
print("code at", ML, "commit", COMMIT)

cands = [Path(p).parent for p in glob.glob("/kaggle/input/**/train", recursive=True)
         if (Path(p).parent / "valid").is_dir() and len(list(Path(p).iterdir())) == 38]
assert cands, "Attach the 'New Plant Diseases Dataset' (vipoooool) as an input."
NPD_ROOT = sorted(cands, key=str)[0]
print("NPD root:", NPD_ROOT)

# PlantDoc: public GitHub, same pinned commit and file layout as the manifests
subprocess.run([sys.executable, "-m", "leafdx.data.download", "--data-dir", "data",
                "--only", "plantdoc"], check=True)
print("PlantDoc files:", sum(1 for _ in Path("data/raw/plantdoc").rglob("*.*")))'''

QUEUES = '''# ---- GPU queues -------------------------------------------------------------------------
import tensorflow as tf, keras
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow", tf.__version__, "| Keras", keras.__version__, "| GPUs:", len(gpus))
assert gpus, "No GPU: enable one under Settings -> Accelerator."
N_GPU = len(gpus)
T_START = time.time()
SKIPPED = []
COMMON = [f"data.npd_root={NPD_ROOT}"]


def pack():
    """Collect finished runs into results_<TAG>.zip (cheap; called after every run)."""
    subprocess.run([sys.executable, "-m", "leafdx.train.report", "collect", str(WORK / "outputs"),
                    "--results", str(WORK / "results")], capture_output=True)
    if (WORK / "logs").exists():
        shutil.copytree(WORK / "logs", WORK / "results" / f"_logs_{TAG}", dirs_exist_ok=True)
    if (WORK / "results").exists():
        subprocess.run(["zip", "-qr", str(WORK / f"results_{TAG}.zip"), "results"], cwd=WORK)


def queues(jobs):
    """Split (cost, args) jobs over the GPUs, longest first."""
    qs, load = [[] for _ in range(N_GPU)], [0] * N_GPU
    for cost, args in sorted(jobs, key=lambda j: -j[0]):
        k = load.index(min(load))
        qs[k].append(args)
        load[k] += cost
    return qs


def run_queue(gpu, jobs, log_path, deadline_h):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TF_CPP_MIN_LOG_LEVEL": "2"}
    with open(log_path, "a") as log:
        for args in jobs:
            if deadline_h is not None and (time.time() - T_START) / 3600 > deadline_h:
                SKIPPED.append(args)
                print(f"[gpu{gpu}] SKIPPED (deadline) {args[1]} {args[-1]}", flush=True)
                continue
            t = time.time()
            log.write(f"\\n>>> {' '.join(args)}\\n")
            log.flush()
            r = subprocess.run([sys.executable, "-m", "leafdx.train.run", *args],
                               env=env, stdout=log, stderr=subprocess.STDOUT)
            status = "OK    " if r.returncode == 0 else "FAILED"
            msg = f"[gpu{gpu}] {status} {args[1]} {args[-1]} in {(time.time() - t) / 60:.1f} min"
            log.write(msg + "\\n")
            log.flush()
            print(msg, flush=True)
            if deadline_h is not None:
                pack()


def run_all(jobs, tag, deadline_h=None):
    logs = WORK / "logs"
    logs.mkdir(exist_ok=True)
    threads = [threading.Thread(target=run_queue,
                                args=(g, q, logs / f"{tag}_gpu{g}.log", deadline_h))
               for g, q in enumerate(queues(jobs))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()'''

SMOKE = '''# ---- smoke pass: every config, a few steps, small evaluations -----------------------------
SMOKE_EVAL = ("eval=[{name: test, manifest: manifests/clean_split/test.csv, limit: 64, "
              "subsets: {original: \\"aug_type == 'original'\\"}}, "
              "{name: plantdoc, manifest: manifests/plantdoc/test.csv, limit: 64, "
              "subsets: {exact: \\"match == 'exact'\\"}}]")
if SMOKE_TEST:
    smoke = WORK / "smoke"
    jobs = [(1, ["--config", c, "--set", *COMMON, f"output_dir={smoke}", "train.epochs=1",
                 "train.steps_per_epoch=5", "train.validation_steps=2", "train.save_model=false",
                 SMOKE_EVAL, "seed=0"]) for c in BATCHES[BATCH]]
    run_all(jobs, "smoke")
    ok = sorted(p.parent.parent.name for p in smoke.glob("*/seed0/metrics.json"))
    print("smoke OK:", ok)
    assert len(ok) == len(BATCHES[BATCH]), "smoke pass failed: see /kaggle/working/logs/smoke_gpu*.log"
    shutil.rmtree(smoke)
    print(f"smoke pass took {(time.time() - T_START) / 60:.1f} min")'''

RUNS = '''# ---- the real runs ------------------------------------------------------------------------
t0 = time.time()
jobs = [(cost, ["--config", c, "--set", *COMMON, f"output_dir={WORK / 'outputs'}", f"seed={s}"])
        for c, cost in BATCHES[BATCH].items() for s in SEEDS]
run_all(jobs, TAG, deadline_h=START_DEADLINE_H)
print(f"runs finished in {(time.time() - t0) / 3600:.2f} h")
print("skipped because of the deadline:", [(a[1], a[-1]) for a in SKIPPED] or "none")'''

PACK = '''# ---- pack results and seed-42 weights ------------------------------------------------------
pack()
models = WORK / "models"
for name in KEEP_MODELS:
    src = WORK / "outputs" / name / "seed42" / "model.keras"
    if src.exists():
        (models / name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, models / name / "model.keras")
        shutil.copy2(src.parent / "config.yaml", models / name / "config.yaml")
if models.exists():
    subprocess.run(["zip", "-qr", str(WORK / f"models_{TAG}.zip"), "models"], cwd=WORK, check=True)
    shutil.rmtree(models)
if (WORK / "outputs").exists():
    shutil.rmtree(WORK / "outputs")  # keeps the saved output small; weights are in models_*.zip
for m in sorted((WORK / "results").glob("p3_*/seed*/metrics.json")):
    d = json.loads(m.read_text())
    e = d["eval"]
    print(f"{d['name']:24s} seed {d['seed']}  test={e['test']['accuracy']:.4f}  "
          f"plantdoc={e['plantdoc']['accuracy']:.4f}  epochs={d['epochs_run']} "
          f"best={d['best_epoch']}  {d['train_seconds'] / 60:.0f} min")
print("download:", sorted(p.name for p in WORK.glob("*.zip")))'''

CELLS = [("markdown", INTRO), ("code", SETTINGS), ("code", SETUP), ("code", QUEUES),
         ("code", SMOKE), ("code", RUNS), ("code", PACK)]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    make = {"markdown": nbf.v4.new_markdown_cell, "code": nbf.v4.new_code_cell}
    nb.cells = [make[kind](src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    return nb


if __name__ == "__main__":
    nbf.write(build(), OUT)
    print("wrote", OUT)
