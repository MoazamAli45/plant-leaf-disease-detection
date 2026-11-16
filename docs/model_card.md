# Model card: LeafDx MobileNetV1 (38 plant leaf disease classes)

The model served by the web app (`web/public/model/`) and the API (`api/model/`). Full
evaluation in [`report.md`](report.md) and [`experiments.md`](experiments.md). Every number
below is copied from the generated tables cited next to it.

## Model details

| | |
|---|---|
| Developer | Syed Moazam Ali (reproduction and extension of Usman & Ali, [`paper.pdf`](paper.pdf)) |
| Date | November 2026, version 0.1.0 |
| Architecture | MobileNetV1 [`howard2017mobilenets`], width 1.0, ImageNet-pretrained, fine-tuned end to end; head: global average pooling, dropout 0.2, dense 38 softmax. 3.27 M parameters |
| Input | RGB image resized to 224 × 224 (bilinear), values 0–255. MobileNet preprocessing (scaling to [−1, 1]) is inside the model |
| Output | probabilities over 38 classes (14 crops; diseases and healthy), order in `web/src/data/classes.json` |
| Training | Adam 1e-4, batch 32, up to 20 epochs, early stopping on validation loss (patience 5), mixed float16, augmentation as model layers (rotation 30°, shift, shear, zoom 0.2, brightness ±20%, flips). Config: `ml/configs/phase3/mobilenet_v1.yaml` |
| Deployed checkpoint | seed 42 of three (`ml/results/p3_mobilenet_v1/seed42/`) |
| Formats | Keras (`inference.keras`, float32, 13.4 MB) for the API; TensorFlow.js graph model with float16 weights (6.6 MB) for the browser |
| Licence | the weights are trained on PlantVillage-derived images (CC BY-SA 3.0); whether weights count as an adaptation under ShareAlike is legally unsettled ([dataset card §1](dataset_card.md#1-datasets-used)). We attribute the datasets. The code licence is stated in the root `README.md` |
| Citation | [`references.bib`](references.bib) |

## Intended use

- **Intended:** research, teaching and demonstration of plant disease classification and of
  its evaluation problems (leakage, field shift).
- **Users:** students, researchers, and people curious about a leaf, who understand that the
  answer may be wrong.
- **Out of scope:** decisions about treatment, pesticide use, sale or destruction of plants;
  any crop or disease outside the 38 classes; photos that are not of a single leaf. The model
  has no "other" class and always answers one of its 38 classes.

## Training data

New Plant Diseases Dataset (Kaggle, version 2), an offline-augmented re-release of
PlantVillage: lab photos of single leaves on uniform backgrounds. Trained on the grouped split
v1 (61,485 train / 13,185 validation images), in which augmented copies and photos of the same
leaf are kept on the same side. See the [dataset card](dataset_card.md).

## Evaluation

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

**Converted formats** (seed 42, 746 leak-free test images, `figures/phase3/efficiency.md`):
Keras 99.06%; TFLite INT8 98.53% (3.55 MB, 1.39 ms median on an Apple M3 Pro CPU). The
TF.js float16 model agrees with Keras on all of 100 verification images
(`figures/phase3/tfjs_verify_mnv1_tfjs_fp16.json`).

## Known limitations and risks

- **Field accuracy is low (16%).** This is the main limitation. Field photos differ from the
  training photos in background, number of leaves, lighting and camera.
- **Confidence is not a reliable warning sign on field photos.** The model is often confidently
  wrong.
- **Residual leakage** in split v1 affects per-class scores for *Raspberry healthy*, *Peach
  healthy*, *Apple cedar apple rust*, *Tomato mosaic virus* and *Potato healthy*
  ([experiments §2](experiments.md#2-a-correction-to-the-clean-split)).
- **Plant-colour gate.** The app's "not a leaf" check counts pixels with plant-like colours;
  green or brown non-leaf objects pass it.
- **Coverage.** 14 crops only, and PlantDoc covers only 28 of the 38 classes, so field
  performance on the other 10 is unknown.

## How the app mitigates these risks

- Verdicts: "not a leaf" (under 12% plant-coloured pixels), "not sure" (top probability under
  0.4), "uncertain" (under 0.7), otherwise the top class. The top three classes are always
  shown.
- Every result states that field accuracy was about 16% in our tests and that the app is not
  agronomic advice.
- Photos are classified in the browser and are not uploaded. The optional Grad-CAM API
  processes images in memory and does not store or log them.

## Ethical considerations

A wrong diagnosis can lead to unnecessary pesticide use or to a disease going untreated. The
app is therefore labelled as a research demonstration and points users to local agricultural
extension services. Disease descriptions in the library (`web/src/data/diseases.json`) are
drafts with cited sources and have not been reviewed by a plant pathologist.
