# Model card: LeafDx MobileNetV1 (38 plant leaf disease classes)

The model served by the web app (`web/public/model/`) and the API (`api/model/`). Full
evaluation in [`report.md`](report.md) and [`experiments.md`](experiments.md). Every number
below is copied from the generated tables cited next to it.

## Model details

| | |
|---|---|
| Developer | Syed Moazam Ali (reproduction and extension of the paper by Ali Usman and Syed Moazam Ali, [`paper.pdf`](paper.pdf)) |
| Date | November 2026, version 0.2.0 (0.1.0: the Phase 3 model before field fine-tuning) |
| Architecture | MobileNetV1 (Howard et al., 2017), width 1.0, ImageNet-pretrained, fine-tuned end to end; head: global average pooling, dropout 0.2, dense 38 softmax. 3.27 M parameters |
| Input | RGB image resized to 224 × 224 (bilinear), values 0–255. MobileNet preprocessing (scaling to [−1, 1]) is inside the model |
| Output | probabilities over 38 classes (14 crops; diseases and healthy), order in `web/src/data/classes.json` |
| Training, stage 1 (lab) | Adam 1e-4, batch 32, up to 20 epochs, early stopping on validation loss (patience 5), mixed float16, augmentation as model layers (rotation 30°, shift, shear, zoom 0.2, brightness ±20%, flips). Config: `ml/configs/phase3/mobilenet_v1.yaml`; seed 42 of three (`ml/results/p3_mobilenet_v1/seed42/`) |
| Training, stage 2 (field) | fine-tuned on all 1,278 PlantDoc field-train photos plus 1,900 lab photos (50 per class) to limit forgetting; Adam 3e-5, batch 32, early stopping on field-val loss (best epoch 13), batch normalisation frozen, random crop / flip / colour augmentation. Protocol: [`field.md`](field.md); run `ml/results/p6_ft_p3_mobilenet_v1_kall_replay50/seed0/` |
| Inference | the probabilities are averaged over 4 views: the photo, its mirror image, an 87.5% centre crop and its mirror image (test-time augmentation), in the browser as in the evaluation |
| Formats | Keras (`inference.keras`, float32) for the API; TensorFlow.js graph model with float16 weights (6.55 MB) for the browser |
| Licence | the weights are trained on PlantVillage-derived images (CC BY-SA 3.0); whether weights count as an adaptation under ShareAlike is legally unsettled ([dataset card §1](dataset_card.md#1-datasets-used)). We attribute the datasets. The code licence is stated in the root `README.md` |
| Citation | see the [report](report.md#references); BibTeX in [`references.bib`](references.bib) |

## Intended use

- **Intended:** research, teaching and demonstration of plant disease classification and of
  its evaluation problems (leakage, field shift).
- **Users:** students, researchers, and people curious about a leaf, who understand that the
  answer may be wrong.
- **Out of scope:** decisions about treatment, pesticide use, sale or destruction of plants;
  any crop or disease outside the 38 classes; photos that are not of a single leaf. The model
  has no "other" class and always answers one of its 38 classes.

## Training data

- **Lab:** New Plant Diseases Dataset (Kaggle, version 2), an offline-augmented re-release of
  PlantVillage: lab photos of single leaves on uniform backgrounds. Trained on the grouped
  split v1 (61,485 train / 13,185 validation images), in which augmented copies and photos of
  the same leaf are kept on the same side.
- **Field:** PlantDoc, split once into field-train (1,278), field-val (381) and field-test (898)
  with near-duplicate copies kept together (`ml/manifests/field_split/`). Only field-train is
  used for training; field-val chose the method and epoch.

See the [dataset card](dataset_card.md).

## Evaluation

### Deployed model (version 0.2.0)

From `figures/phase6/deployed.json`, with the 4-view test-time augmentation the app uses:

| Test set | Before field fine-tuning | **Deployed** |
|---|---|---|
| Field: PlantDoc field-test (898 images, 28 classes) | 18.15% | **54.34%** |
| Field-test macro-F1 / crop accuracy / ECE | | 48.70% / 73.16% / 7.01% |
| Lab, leak-free test (11,599 images) | 98.55% | 97.78% (ECE 1.20%) |

Over three fine-tuning seeds the method scores 53.08 ± 1.28% on field-test. The deployed seed
was the best of three on field-val, so 53% is the better estimate. Per class
(`figures/phase6/per_class_deployed.md`): 22 of 28 field classes improve; *Corn: Cercospora /
Gray leaf spot*, *Tomato: Early blight*, *Tomato: Late blight*, *Potato: Early blight* and
*Pepper: Bacterial spot* get worse, mostly confused with similar diseases of the same crop.

The TF.js model agrees with Keras on all 100 verification images (largest probability
difference 0.012), and the app's test-time augmentation agrees with the Python evaluation on
all 100 (`figures/phase6/tfjs_verify*.json`).

### Before field fine-tuning (version 0.1.0, Phase 3)

Mean ± std over three seeds (`figures/phase3/main_results.md`):

| Test set | Accuracy (%) | Macro-F1 (%) | ECE (%) |
|---|---|---|---|
| Lab, held-out test (13,171 images) | 98.40 ± 0.38 | 98.39 ± 0.38 | 0.60 ± 0.24 |
| Lab, leak-free test subset (11,599 images) | 98.26 ± 0.43 | | |
| **Field: PlantDoc (2,557 images, 28 classes)** | **16.33 ± 1.12** | 15.60 ± 0.96 | 62.18 ± 2.25 |
| Field: PlantDoc, exact-match classes | 22.23 ± 1.58 | | |

On field images (`figures/phase3/field_analysis.md`) the model names the right crop for 37.48%
of photos, and its wrong answers average 77% confidence. At a 0.9 confidence threshold it still
answers 39.5% of field images and is right on 25.1% of those
(`figures/phase3/selective_prediction.md`).

**Converted formats** (Phase 3 model, seed 42, 746 leak-free test images, `figures/phase3/efficiency.md`):
Keras 99.06%; TFLite INT8 98.53% (3.55 MB, 1.39 ms median on an Apple M3 Pro CPU). The
TF.js float16 model agrees with Keras on all of 100 verification images
(`figures/phase3/tfjs_verify_mnv1_tfjs_fp16.json`).

## Known limitations and risks

- **Field accuracy is moderate (54% on PlantDoc field-test).** About half of field photos get
  the wrong disease, although the crop is right for 73%. Before field fine-tuning it was 16–18%.
- **Field results come from one source.** Fine-tuning and testing both used PlantDoc, a set of
  web-collected images, many of them stock photos. Photos from other phones, regions or seasons
  will probably score lower; this has not been measured.
- **Look-alike diseases.** Diseases with similar lesions on the same crop (blights and leaf
  spots of tomato and potato; corn Cercospora vs Northern Leaf Blight) are often confused.
- **Confidence on field photos.** Calibration error fell from 60% to 7% after field
  fine-tuning, so confidence is now informative on average, but individual confident answers
  can still be wrong.
- **Residual leakage** in split v1 affects per-class scores for *Raspberry healthy*, *Peach
  healthy*, *Apple cedar apple rust*, *Tomato mosaic virus* and *Potato healthy*
  ([experiments §2](experiments.md#2-a-correction-to-the-clean-split)).
- **Plant-colour gate.** The app's "not a leaf" check counts pixels with plant-like colours;
  green or brown non-leaf objects pass it.
- **Coverage.** 14 crops only, and PlantDoc covers only 28 of the 38 classes, so field
  performance on the other 10 is unknown, and those 10 classes saw no field photos in training.

## How the app mitigates these risks

- Verdicts: "not a leaf" (under 12% plant-coloured pixels), "not sure" (top probability under
  0.4), "uncertain" (under 0.7), otherwise the top class. The top three classes are always
  shown.
- Every result states the measured field accuracy (54%) and that the app is not agronomic
  advice.
- Photos are classified, and Grad-CAM heat maps computed, in the browser; they are never
  uploaded. (The separate API in `api/`, not used by the app, processes images in memory and
  does not store or log them.)

## Ethical considerations

A wrong diagnosis can lead to unnecessary pesticide use or to a disease going untreated. The
app is therefore labelled as a research demonstration and points users to local agricultural
extension services. Disease descriptions in the library (`web/src/data/diseases.json`) are
drafts with cited sources and have not been reviewed by a plant pathologist.
