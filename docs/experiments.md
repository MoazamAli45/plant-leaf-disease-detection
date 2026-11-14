# Phase 3: corrected and extended experiments

> **Status:** main comparison done for 4 of 5 models (3 seeds each). MobileNetV3-Small,
> the ablations and the efficiency study are pending, and are marked ⟨PLACEHOLDER⟩ below.
> Every number here is generated from `ml/results/` by `python -m leafdx.train.report_phase3`
> (tables in [`figures/phase3/`](figures/phase3/)), `leafdx.explain.gradcam` and
> `leafdx.train.efficiency`.

## 1. What changed from the reproduction

| | Reproduction (Phase 2) | Phase 3 |
|---|---|---|
| Split | paper 80/20 (90.3% of validation images linked to train) | clean grouped split v1, 70/15/15 (see §2 for its residual leakage) |
| Reported on | validation set | held-out **test** set, plus **PlantDoc** field images |
| CNN output | sigmoid | **softmax** |
| Backbones | MobileNetV1 (frozen) | MobileNetV1, V2, V3-Small, EfficientNetB0, all **fine-tuned** |
| Preprocessing | `rescale=1/255` for all, nearest resize | each backbone's own ImageNet preprocessing, bilinear resize |
| Augmentation | same transforms and values (A5), on the CPU | same transforms and values, as model layers on the GPU |
| Epochs | 20, last epoch | up to 20, early stopping (patience 5) on validation loss, best epoch kept |
| Optimiser | Adam 1e-3 | Adam 1e-3 (CNN) and 1e-4 (fine-tuned backbones) |
| Precision | float32 | mixed float16 |
| Seeds | 1 | 3 (42, 43, 44); mean ± std reported |

All runs were trained on Kaggle (2 × Tesla T4, TensorFlow 2.21, Keras 3.13) at git commit
`f55ca40`. Logs: `ml/results/_logs_kaggle_v1/`.

## 2. A correction to the clean split

While preparing this analysis we found that NPD also contains an offline augmentation named
`_newPixel25` (2,399 images), which the Phase 1 file-name parser did not recognise. Those
copies were therefore not grouped with their source photos. With corrected parsing
(`python -m leafdx.data.recheck`, `figures/data_audit/recheck_newpixel.json`):

| Split | Images linked to training by source photo, leaf or file |
|---|---|
| paper 80/20, validation | **90.31%** (15,869 of 17,572); Phase 1 reported 87.5% |
| clean split v1, validation | 11.81% (1,557 of 13,185) |
| clean split v1, test | 11.94% (1,572 of 13,171) |

The links are **concentrated in a few classes**: 98.2% of *Raspberry healthy* test images
(6 leak-free images remain), 93.8% of *Peach healthy*, 63.0% of *Apple cedar apple rust*,
53.0% of *Tomato mosaic virus* and 49.1% of *Potato healthy*. Per-class test scores for these
classes are therefore not reliable on split v1, and split v2 would be needed for them.

The models were already trained on v1, so we keep v1. Test results are reported both on the
full test set and on the **leak-free subset** (the 11,599 test images with no link). The
difference is at most 0.15 pp for every model (table below), so the conclusions do not
depend on it. The parser is fixed. `parse_newpixel_suffix: false` in `ml/configs/data.yaml`
keeps v1 reproducible byte for byte, and setting it to `true` produces a corrected split v2.

## 3. Main results (mean ± std over 3 seeds)

Full table: [`figures/phase3/main_results.md`](figures/phase3/main_results.md).

| Model | Params (M) | Test acc. (%) | Test acc., leak-free (%) | Test macro-F1 (%) | Test ECE (%) | PlantDoc acc. (%) | PlantDoc macro-F1 (%) | PlantDoc acc., exact-match classes (%) |
|---|---|---|---|---|---|---|---|---|
| Paper CNN (softmax) | 2.43 | 88.40 ± 0.45 | 88.46 ± 0.38 | 88.39 ± 0.41 | 2.84 ± 0.20 | 7.46 ± 1.02 | 6.72 ± 1.23 | 9.28 ± 1.64 |
| MobileNetV1 | 3.27 | 98.40 ± 0.38 | 98.26 ± 0.43 | 98.39 ± 0.38 | 0.60 ± 0.24 | 16.33 ± 1.12 | 15.60 ± 0.96 | 22.23 ± 1.58 |
| MobileNetV2 | 2.31 | 97.43 ± 0.40 | 97.47 ± 0.43 | 97.40 ± 0.38 | 1.16 ± 0.31 | 15.02 ± 1.98 | 13.75 ± 1.11 | 19.06 ± 2.58 |
| MobileNetV3-Small | ⟨PLACEHOLDER⟩ | | | | | | | |
| **EfficientNetB0** | 4.10 | **99.22 ± 0.09** | **99.18 ± 0.07** | 99.21 ± 0.08 | 0.32 ± 0.03 | **18.15 ± 0.58** | 17.76 ± 0.32 | 22.23 ± 1.37 |

- **Lab test.** On held-out lab images, fine-tuned backbones reach 97–99%. The paper's CNN,
  trained from scratch, reaches 88.4%.
- **Field images.** Every model collapses on PlantDoc: 7–18% accuracy, against 2.6% chance.
  The gap between leak-free lab accuracy and field accuracy is **81–82 pp for every model**.
  A better lab score does not close it.
- **Per class.** Precision, recall and F1 per class (mean ± std over seeds), for every model
  and both test sets, are in `figures/phase3/per_class/`. Confusion matrices (seed 42) are in
  `figures/phase3/confusion/`.

## 4. Why field accuracy is so low

From [`figures/phase3/field_analysis.md`](figures/phase3/field_analysis.md):

| Model | PlantDoc acc. | Correct crop (any disease) | Healthy vs diseased (baseline 66.8%) | Predictions on 4 most-predicted classes | Predictions on classes absent from PlantDoc | Mean confidence: correct / wrong |
|---|---|---|---|---|---|---|
| Paper CNN | 7.46% | 23.90% | 72.27% | 38.52% | 34.57% | 79.9% / 75.4% |
| MobileNetV1 | 16.33% | 37.48% | 76.50% | 43.23% | 26.09% | 86.0% / 77.1% |
| MobileNetV2 | 15.02% | 32.24% | 72.44% | 41.75% | 21.84% | 84.2% / 76.6% |
| EfficientNetB0 | 18.15% | 38.74% | 71.88% | 43.49% | 24.66% | 83.6% / 73.7% |

1. **Even the crop is mostly wrong.** The best model names the right plant species only 39%
   of the time. Healthy-versus-diseased is barely above always answering "diseased".
2. **Predictions collapse.** 39–43% of field predictions fall on four classes, and 22–35%
   fall on diseases that do not occur in PlantDoc at all (e.g. *Strawberry leaf scorch*,
   `field_predictions_p3_efficientnet_b0.png`).
3. **The models are confidently wrong.** Calibration error is 57–68% on PlantDoc, against at
   most 2.8% on the lab test set. Wrong answers average 74–77% confidence.
4. **A confidence threshold does not rescue it.** At a 0.9 threshold, EfficientNetB0 still
   answers 44.9% of field images, and only 23.2% of those answers are right; on lab images it
   is 99.8% right at 98.4% coverage (`selective_prediction.md`, `selective_*.png`). The
   web app's "not sure" response therefore cannot be relied on to catch field errors, and the
   app must say so.

## 5. What do the models look at?

Seed-42 models, 185 clean-test originals (5 per class; 5 images had no upstream
segmentation). Leaf masks are the PlantVillage authors' segmentations. Chance = 2.6%.
Source: `figures/phase3/explain/summary.json`.

| Model | Accuracy | Grad-CAM leaf ratio (median) | Leaf only (background grey) | Background only | Border strip only |
|---|---|---|---|---|---|
| Paper CNN | 85.9% | 1.07 | 51.4% | 13.0% | 14.6% |
| MobileNetV1 | 97.3% | 1.46 | 89.2% | 7.6% | 6.5% |
| MobileNetV2 | 97.8% | 1.60 | 87.0% | 4.9% | 5.9% |
| EfficientNetB0 | 98.9% | 1.82 | 71.9% | 8.1% | 11.9% |

- **Fine-tuned backbones attend to the leaf.** 62–83% of their Grad-CAM mass is on the leaf,
  which covers 47% of the image, and they keep 72–89% accuracy when the background is
  removed. In the examples (`explain/gradcam_lab.png`), the heat maps sit on the lesions:
  corn blight streaks, cedar apple rust spots, target spot.
- **The paper CNN barely prefers the leaf** (ratio 1.07), and in one example it highlights
  the background around the leaf.
- **Background cues exist but are weaker than reported elsewhere.** From the background
  alone, or from a 16-pixel border strip with no leaf shape, models score 5–15%, i.e. 2–6×
  chance. Noyan (2022) reports 49% for a model *trained* on 8 background pixels. Our models
  were not trained on backgrounds, so the two numbers measure different things. Caveat: the
  border strip still contains leaf pixels when a leaf fills the frame (e.g. corn).
- **On field images** (`explain/gradcam_field.png`), attention spreads over soil,
  neighbouring leaves and stock-photo watermarks, which fits the field results in §4.

An earlier version of this analysis used saturation-based leaf masks. Inspection showed them
to be wrong for dark and powdery leaves, so those numbers were discarded before publication.

## 6. Ablations (MobileNetV1, 3 seeds)

⟨PLACEHOLDER: augmentation off, frozen backbone (the paper's set-up), width multiplier 0.75
and 0.5. Pending Kaggle runs. Table: `figures/phase3/ablations.md`.⟩

## 7. Efficiency

Seed-42 models on an Apple M3 Pro CPU: batch 1, 4 threads, median of 50 runs after 10
warm-up runs. Accuracy is measured on 746 leak-free clean-test images (up to 20 per class;
*Raspberry healthy* has only 6). INT8 means full integer quantisation (weights and
activations), calibrated on 200 training images, with float input and output. Full table:
[`figures/phase3/efficiency.md`](figures/phase3/efficiency.md). MobileNetV3-Small will be
added when its runs exist.

| Model | Format | Size (MB) | CPU latency (ms) | Accuracy (%) | Agreement with Keras (%) |
|---|---|---|---|---|---|
| Paper CNN | Keras float32 | 9.76 | 2.68 | 88.20 | 100 |
| | TFLite INT8 | 2.45 | 0.57 | 87.80 | 98.66 |
| MobileNetV1 | Keras float32 | 13.40 | 7.30 | 99.06 | 100 |
| | TFLite float32 | 12.96 | 3.93 | 99.06 | 100 |
| | TFLite dynamic-range | 3.42 | 2.84 | 99.06 | 99.87 |
| | **TFLite INT8** | **3.55** | **1.38** | **98.53** | 99.20 |
| | TF.js uint8 weights | 3.32 | (browser, Phase 4) | | |
| MobileNetV2 | Keras float32 | 9.82 | 7.47 | 97.72 | 100 |
| | TFLite INT8 | 2.76 | 1.34 | 95.58 | 97.05 |
| EfficientNetB0 | Keras float32 | 17.24 | 11.18 | 99.60 | 100 |
| | TFLite dynamic-range | 4.60 | 6.71 | 99.46 | 99.60 |
| | TFLite INT8 | 4.96 | 2.27 | **69.84** | 69.97 |

- **MobileNetV1 quantises almost losslessly.** Full INT8 is 3.8× smaller and 5.3× faster
  than the Keras model, for −0.53 pp, so it is the model deployed in the browser (Phase 4).
- **EfficientNetB0 does not survive full INT8 post-training quantisation** (−29.8 pp). Its
  swish activations and squeeze-and-excitation blocks are known to be sensitive to
  activation quantisation. Its dynamic-range version (int8 weights, float activations)
  keeps 99.46%. Quantisation-aware training was not tried.
- **MobileNetV2 loses 2.1 pp** under full INT8.
- **Accuracy after conversion was measured, not assumed.** The agreement column compares
  each converted model's predictions with its Keras original.

Environment: TensorFlow 2.20 for TFLite; TF.js converter 4.22 in a separate environment
(`ml/requirements-tfjs.txt`).
