# Project context: Plant leaf disease detection (LeafDx)

Snapshot taken 2026-11-20. Paste this into a new Claude conversation as background. Every number
below comes from the repository's saved results (`ml/results/`, `docs/figures/`) or its docs.

---

## 1. What the project is

A research + engineering project that classifies plant leaf diseases from photos (38 classes,
14 crops, healthy or diseased) and asks a harder question than "how accurate is it?": **does lab
accuracy hold on real field photos?**

- **Authors:** Syed Moazam Ali (GitHub `MoazamAli45`) and Ali Usman. The technical report
  (`docs/report.md`) is by Syed Moazam Ali; the original paper it revisits is by Ali Usman and
  Syed Moazam Ali.
- **Origin:** an earlier unpublished paper of theirs (UET Taxila, Dept. of Software Engineering),
  *"Plant Leaf Disease Detection Using Traditional CNN and MobileNet Architecture"*
  (`docs/paper.pdf`). It reported a custom CNN at 90.25% and MobileNet at 95.47% validation
  accuracy on the Kaggle *New Plant Diseases Dataset* (NPD).
- **Repo:** <https://github.com/MoazamAli45/plant-leaf-disease-detection>
- **Core story:** (1) the standard NPD split leaks badly, (2) on truly unseen lab images networks
  are 97.5–99.2% accurate, (3) on field photos (PlantDoc) every model collapses to 7–18% and is
  confidently wrong, (4) labelled field photos are what helps: fine-tuning lifts MobileNetV1 from
  18% to ~54% on held-out field photos, (5) the model ships in an offline, in-browser web app
  that is honest about its field accuracy.
- **Disclaimer used everywhere:** research project, not agronomic advice.

## 2. Repository layout

| Folder | Contents |
|---|---|
| `ml/` | Python package `leafdx`: data download/audit/split, training, evaluation, Grad-CAM, compression, field adaptation. `configs/` (YAML), `manifests/` (split CSVs with SHA-256), `results/` (per-run `metrics.json`, probabilities), `models/`, `notebooks/` (Kaggle notebook) |
| `ml/src/leafdx/data/` | `audit.py`, `hashing.py` (pHash, D4-invariant), `split.py`, `plantdoc.py`, `field_split.py`, `fieldplant.py`, `extra_field.py` (FieldPlant, CD&S), `classes.py`, etc. |
| `ml/src/leafdx/train/` | `run.py`, `models.py`, `data.py`, `metrics.py`, `report*.py`, `efficiency.py`, `export_web.py` |
| `ml/src/leafdx/explain/` | `gradcam.py` |
| `ml/src/leafdx/field/` | Phase 6/7 field adaptation: `run.py`, `methods.py`, `report.py`, `deploy.py`, `phase7.py` |
| `web/` | Next.js 16 / React 19 / TypeScript / Tailwind 4 / shadcn PWA with TensorFlow.js 4.22. Pages: `/` (home), `/detect`, `/result`, `/library` + `/library/[slug]`, `/research`, `/about`. Libs: `model.ts`, `tta.ts`, `gradcam.ts`, `verdict.ts`, `classes.ts`, `research.ts`, `storage.ts`. Model in `web/public/model/` |
| `api/` | Optional FastAPI service (`leafdx_api`): `GET /health`, `GET /classes`, `POST /predict` (`?gradcam=true`). Rate-limited, CORS, images never stored. Not used by the app |
| `docs/` | `report.md` (main paper), `field.md` (Phase 6), `field2.md` (Phase 7), `model_card.md`, `dataset_card.md`, `experiments.md`, `reproduction.md`, `reproducing.md`, `deployment.md`, `references.bib`, `figures/` |
| `.github/workflows/ci.yml` | ml + api: ruff check, ruff format --check, pytest. web: lint, typecheck, vitest, build, Playwright e2e |

**Stack:** Python 3.12, TensorFlow/Keras 3 (TF 2.21 on Kaggle), NumPy, pandas, scikit-learn;
TFLite, TensorFlow.js (separate `.venv-tfjs`); Next.js, Vitest, Playwright; FastAPI; ruff, ESLint.
Training ran on free Kaggle GPUs (2× Tesla T4); later phases also run locally (Apple M3 Pro).

## 3. Data

| Dataset | Size | Role | Licence |
|---|---|---|---|
| New Plant Diseases Dataset v2 (NPD, Kaggle re-release of PlantVillage) | 87,867 images (70,295 train / 17,572 val), 38 classes, 14 crops; 48,239 originals, rest offline-augmented copies | lab training/testing | CC BY-SA 3.0 |
| PlantDoc | 2,578 web-collected images → 2,557 after cleaning; covers 28 of the 38 classes via hand-written mapping (`ml/resources/plantdoc_to_plantvillage.csv`) | field test (Phase 3), field train/val/test (Phase 6+) | CC BY 4.0 |
| FieldPlant (Moupojou et al., 2023) | 5,156 plantation photos (Cameroon); leaf boxes cropped → 7 usable classes (corn NLB, gray leaf spot, common rust, healthy; tomato healthy, YLCV, mosaic). 71% of crops are corn NLB | Phase 7 train/val/test | CC BY 4.0 |
| CD&S (Ahmad et al., 2021) | 1,020 corn photos (gray leaf spot, NLB), USA | Phase 7 **test only** (no licence stated, not redistributed) | none stated |
| Plant Pathology 2020 | — | **not used** (terms not accepted/checked) | — |

The repo contains no dataset images except small attributed example figures.

## 4. Phases and results

All results are mean ± std over 3 seeds unless stated.

### Phase 1 — Leakage audit
Images linked by: augmentation suffixes in NPD file names (`_flipLR`, `_180deg`, `_newPixel25`, …),
PlantVillage upstream leaf IDs (`leaf-map.json`, 7,378 leaves, 69.2% of images), SHA-256, and a
64-bit DCT pHash made invariant to the 8 rotations/mirrors (0-bit matches only, within class).

- **90.31% of the standard validation images (15,869 / 17,572) are linked to a training image.**
  (First audit said 87.5%; the `_newPixel25` pattern, 2,399 images, was missed and later fixed.)
- Grouped 70/15/15 split ("split v1"): 22,133 groups → 61,485 / 13,185 / 13,171 images.
  Because of the missed pattern, 11.94% of v1 test images are still linked to training, so all
  results are also reported on the **leak-free test subset of 11,599 images**; the two differ by
  ≤ 0.15 pp. A corrected split v2 can be generated but has not been used for training.

### Phase 2 — Reproduction of the original paper (seed 42, paper's leaky split)
Nine unstated settings recorded as assumptions A1–A9 (`docs/reproduction.md`): categorical CE,
Adam 1e-3, batch 32, frozen MobileNet backbone (only 38,950-param head trained), full-epoch schedule.

| Model | Paper val acc | Reproduced | Δ |
|---|---|---|---|
| CNN (sigmoid) | 90.25 | 84.58 | −5.67 pp |
| MobileNet (frozen) | 95.47 | 94.88 | −0.59 pp |

The "100 steps/epoch" reading gave 45.67% for the CNN and is kept as a documented negative result.

### Phase 3 — Five models on the clean split, plus field test
Fine-tuned ImageNet backbones (GAP → dropout 0.2 → 38-way softmax), Adam 1e-4 (CNN 1e-3),
batch 32, ≤ 20 epochs, early stopping (patience 5), mixed fp16, seeds 42/43/44.

| Model | Params (M) | Leak-free lab test | Lab ECE | PlantDoc (2,557) | PlantDoc ECE |
|---|---|---|---|---|---|
| Paper CNN (softmax) | 2.43 | 88.46 ± 0.38 | 2.84 | 7.46 ± 1.02 | 68.24 |
| MobileNetV1 | 3.27 | 98.26 ± 0.43 | 0.60 | 16.33 ± 1.12 | 62.18 |
| MobileNetV2 | 2.31 | 97.47 ± 0.43 | 1.16 | 15.02 ± 1.98 | 62.72 |
| MobileNetV3-Small | 0.96 | 98.45 ± 0.18 | 0.55 | 13.84 ± 1.22 | 60.42 |
| EfficientNetB0 | 4.10 | **99.18 ± 0.07** | 0.32 | **18.15 ± 0.58** | 57.43 |

- Lab → field gap: **81–85 pp** for every model. Chance = 2.6%.
- Field failure analysis: right crop only 24–39% of the time; predictions collapse onto 4 classes
  (39–44%) and 22–35% onto classes absent from PlantDoc; wrong answers average 73–77% confidence,
  so confidence thresholds don't help (EffB0 at 0.9 threshold: 44.9% coverage, 23.2% accurate).
- Leakage inflated lab accuracy only ~1.3 pp (frozen ablation 93.58% leak-free vs 94.88% leaky):
  **the field shift, not leakage, is what makes lab numbers misleading.**
- **Grad-CAM / counterfactuals** (185 images): backbones focus on the leaf (leaf ratio 1.46–1.82;
  CNN 1.07), keep 72–89% accuracy with background removed, 5–15% from background alone. On field
  photos attention spreads to soil, neighbouring leaves, watermarks.
- **Ablations (MobileNetV1, Welch t-test):** no augmentation → +1.04 pp lab (p = 0.048), no field
  change; frozen backbone → −4.68 pp lab (p = 0.0002); width 0.75 / 0.5 keep lab accuracy, field
  differences (~−2 pp) not significant.
- **Efficiency (seed 42, M3 Pro CPU):** MobileNetV1 Keras 13.4 MB / 7.34 ms / 99.06% → TFLite INT8
  3.55 MB / 1.39 ms / 98.53% (−0.54 pp). Width-0.5 INT8: 1.00 MB, 98.12%. Full INT8 **breaks**
  EfficientNetB0 (69.84%) and MobileNetV3-Small (60.86%); dynamic-range keeps both > 99%.

### Phase 4–5 — App, API, docs
See §5 below. Report, model card, dataset card written; Docker was dropped; deployment is Vercel.

### Phase 6 — Closing the field gap (protocol pre-registered 2026-11-18, `docs/field.md`)
PlantDoc split once, grouped by near-duplicates (D4 pHash ≤ 6 bits), stratified 50/15/35, seed 42:
**field-train 1,278 / field-val 381 / field-test 898**. All choices on field-val; field-test reported
once. "Lab check" = 746 leak-free lab test images (≤ 20/class) to measure forgetting. Decision
rule: lab check within 2 pp of baseline, MobileNetV1-sized.

| MobileNetV1 method | Field-test acc | Field-test ECE | Lab check |
|---|---|---|---|
| Baseline (Phase 3 seed 42) | 18.15 | 59.73 | 99.06 |
| TTA (4 views: orig, hflip, 87.5% centre crop, its flip) | 20.04 | 51.06 | 99.20 |
| AdaBN (unlabelled field-train) | 19.78 ± 0.13 | 56.20 | 93.16 |
| Fine-tune k=5 / 10 / 20 per class (20 lab replay/class) | 26.58 / 34.19 / 40.76 | 13–15 | ~98 |
| Fine-tune all (~45/class), 20 replay | 49.33 ± 1.78 | 9.24 | 96.38 (ineligible) |
| Fine-tune all, 50 replay | 50.00 ± 0.68 | 9.91 | 97.94 |
| **Fine-tune all, 50 replay + TTA (chosen)** | **53.08 ± 1.28** | **7.24** | 98.26 |

Fine-tune recipe: Adam 3e-5, batch 32, ≤ 15 epochs, early stop on field-val loss (patience 3),
BN in inference mode, random resized crop + flips + colour jitter, seeds 0/1/2.

**Deployed model** = seed 0 of the chosen method (best field-val, 54.59%):
- Field-test **54.34%** (honest estimate: method mean 53.08 ± 1.28), macro-F1 48.70%, crop
  accuracy 73.16%, ECE 7.01% (was 59.73%). Without TTA: 50.33%.
- Full leak-free lab test (11,599): **97.78%** (was 98.55%), ECE 1.20%.
- TF.js float16: 6.55 MB. 22/28 classes improve; 5 get worse (similar-looking diseases of the same
  crop, e.g. corn gray leaf spot 83 → 22%, mostly now called NLB).
- Browser checks: TF.js = Keras top-1 on 100/100 images (max prob diff 0.012); app TTA = Python
  TTA 100/100 (0.008); in-browser Grad-CAM median correlation 0.99998; ~29 ms/view (WASM, Node).
- Deviations logged: lab replay raised 20 → 50 (lr check lost 2.55 pp); seed choice rule added.
- EfficientNetB0 k=5 fine-tune: 31.48 ± 0.82% field-test (comparison only).

### Phase 7 — More field data, larger model, independent test (IN PROGRESS, `docs/field2.md`)
Protocol fixed 2026-11-18; deviation added 2026-11-19 (overfitting checks: train–val gap, best
epoch, loss curves, 1,000-resample bootstrap CIs; "no lucky seed" = must beat Phase 6 on the
**mean** selection score; tie → smaller train–val gap, then smaller model). §6 Results still says
*Pending*.

- **Methods:** P = PlantDoc field-train only; PF = PlantDoc + FieldPlant-train capped at 100
  crops/class (390 crops); PF-long only if best epoch ≥ 13 for ≥ 2 seeds. Both MobileNetV1 and
  EfficientNetB0 (~8 MB in browser, now allowed). Early stopping on pooled PlantDoc-val +
  FieldPlant-val loss. 50 lab replay/class, seeds 0/1/2.
- **Selection score** = mean(PlantDoc-val accuracy, FieldPlant-val balanced accuracy).
- **Deploy only if:** lab check within 2 pp of its own start, TF.js fp16 ≤ 10 MB, passes browser
  checks (≥ 98% top-1 agreement incl. TTA and Grad-CAM), beats Phase 6 deployed on selection score.
- FieldPlant split: 1,159 / 349 / 812 crops. CD&S: 1,020 images, test only.

**Results so far** (my summary computed from `ml/results/p7_*/seed*/metrics.json`; not yet in a
report; "sel" = selection score; all with TTA unless noted):

| Run | Seeds | Sel (val) | PlantDoc-test acc | FieldPlant-test acc | FieldPlant-test bal. acc | Lab check |
|---|---|---|---|---|---|---|
| Phase 6 deployed (reference) | 1 | 42.11 | 54.34 | 50.00 | 24.85 | 97.99 |
| P, MobileNetV1 | 3 | 38.26 ± 1.58 | 50.59 ± 1.27 | 62.81 ± 2.59 | 27.74 ± 2.18 | 98.30 ± 0.56 |
| **PF, MobileNetV1** | 3 | **54.89 ± 1.05** | 50.59 ± 1.83 | **79.72 ± 2.10** | **61.78 ± 0.89** | 98.53 ± 0.70 |
| P, EfficientNetB0 | 3 | 39.50 ± 0.25 | **58.87 ± 0.68** | 66.54 ± 0.43 | 27.94 ± 1.89 | 99.33 ± 0.13 |
| PF, EfficientNetB0 | — | running | | | | |

Unadapted references: Phase 3 MobileNetV1 FieldPlant-test 15.15% (TTA), EfficientNetB0 20.81%.
PF best epochs (MNV1): 7, 12, 11, so PF-long is probably not triggered for MNV1.

**Status at snapshot:** `leafdx.field.phase7 run --model p3_efficientnet_b0 --method PF --seed 0`
and the CD&S download (`leafdx.data.extra_field cds`) are running. `ml/results/_logs_p7/finish.sh`
waits for both, then runs `phase7 add-cds`, `phase7 diagnose`, `phase7 report` → fills
`docs/figures/phase7/` and docs/field2.md §6. No CD&S scores exist yet. The deployment decision
for Phase 7 has not been made.

## 5. The web app (`web/`)

- In-browser MobileNetV1 (currently the **Phase 6 fine-tuned** model), TF.js GraphModel, float16
  weights, 224×224 raw RGB in [0, 255] (preprocessing inside the model). Backend order: WebGL →
  WASM → CPU. Model metadata in `/model/leafdx.json` (`ModelInfo`). Photos never leave the device.
- PWA with service worker (`public/sw.js`): works offline after first visit; installable.
- 4-view TTA in `lib/tta.ts`, batched in one call.
- Verdicts (`lib/verdict.ts`): "not a leaf" if plant-coloured pixel share < 0.12 (hue 20–170°,
  heuristic), "not sure" if top prob < 0.40, "uncertain" < 0.70, else "confident". Always shows
  top-3 and states measured field accuracy (54% on held-out PlantDoc).
- **Grad-CAM on device without autodiff** (`lib/gradcam.ts`): since the net ends in GAP + dense
  softmax, ∂p_c/∂A_k(u,v) = (p_c / HW)(W_kc − Σ_j p_j W_kj), constant over positions, so one
  forward pass (feature maps + probs) plus dense weights gives the map. Matches TF gradient
  Grad-CAM: median Pearson 0.99997 (min 0.997), same hottest 7×7 cell on 100/100.
- Pages: home, detect, result, 38-class library, research results (`data/research.json`), about.
- Deployment: Vercel (root dir `web`, free). API optional on Hugging Face Spaces (Docker; may need
  PRO as of mid-2026).
- Why MobileNetV1 and not EffB0/V3-Small: it's the original paper's model, quantises cleanly,
  and beats V3-Small in the field.

## 6. Key commands

```bash
cd ml && make setup PYTHON=python3.12 && make data && make prepare && make eda
.venv/bin/python -m leafdx.train.run --config configs/phase3/mobilenet_v1.yaml --set seed=43
.venv/bin/python -m leafdx.train.report_phase3 --results results --out ../docs/figures/phase3
.venv/bin/python -m leafdx.data.field_split
.venv/bin/python -m leafdx.field.run finetune --model p3_mobilenet_v1 --seed 0
.venv/bin/python -m leafdx.field.report
.venv/bin/python -m leafdx.field.deploy --run <run>/seed0 --tta
.venv/bin/python -m leafdx.field.phase7 {reference|run|add-cds|diagnose|report}
.venv/bin/python -m leafdx.train.export_web {classes|research|api-model|verify}
cd web && npm install && npm run dev        # also: lint, typecheck, test, e2e, verify-model
cd api && .venv/bin/uvicorn leafdx_api.main:app --port 8000
make test / make lint  (ml)
```

## 7. Working state and conventions

- Branch `main`. Last commit: `3e4dd8e Author name for research paper`. **Large uncommitted
  work:** Phase 6 docs/report/model card updates, the new Phase 6 model in `web/public/model/`,
  app changes (TTA, `leafdx.json` model info, research page), `docs/field.md`, `docs/field2.md`,
  new manifests (`field_split/`, `fieldplant/`), and all `ml/results/p6_*` and `p7_*` runs.
- **The user commits and pushes themselves.** Don't run `git commit`/`git push`.
- **No AI attribution** in commits/PRs (no `Co-Authored-By: Claude`).
- House style: every reported number must come from a script reading saved results, with the
  source file named next to each table; protocols are fixed before experiments and any change is
  logged as a dated deviation; test sets are never used for choices; report mean ± std over seeds
  and the honest mean next to any best-seed number; plain, direct English.

## 8. Known limitations / open items

- Split v1 has residual leakage (11.94% of test linked); per-class scores unreliable for ~5
  classes; v2 split not yet used.
- PlantDoc is small and noisy (watermarks, stock photos on white backgrounds, likely label errors);
  field-test has 898 images, one class (tomato spider mites) with 1 test image.
- Phase 6 trains and tests on the same web source, so 54% is an upper estimate for other phones /
  regions. Phase 7's CD&S test is meant to address this.
- Only 3 seeds: little power for 1–2 pp differences. Latency measured on a laptop, not a phone.
- Not tried: background randomisation, leaf segmentation, quantisation-aware training.
- To finish Phase 7: wait for EffB0 PF runs + CD&S, run diagnose/report, apply the deployment
  rule, then (if a new model wins) redeploy to `web/public/model`, re-verify in the browser, and
  update report, model card, README and the app's stated field accuracy.
