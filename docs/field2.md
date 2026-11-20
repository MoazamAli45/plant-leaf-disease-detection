# Phase 7: more field data, a larger model, and a test from an independent source

> **Status:** protocol fixed on 2026-11-18, before any Phase 7 model was trained or evaluated.
> Results (§6) are taken from `ml/results/p7_*/` by `python -m leafdx.field.phase7 report`.
> Any later change to this protocol is listed in §7 with its date.

## 1. Goal

Phase 6 ([`field.md`](field.md)) raised accuracy on held-out PlantDoc photos from 18% to 54% by
fine-tuning MobileNetV1 on about 45 labelled PlantDoc photos per class. Its learning curve had
not levelled off, and both training and test photos came from one web-collected source.
Phase 7 asks:

1. Does **more labelled field data** from a second source (photos taken in plantations) raise
   accuracy further?
2. Does a **larger network** (EfficientNetB0, about 8 MB in the browser) help, now that the
   app may use it?
3. How well do the models do on **photos from a source never used for training**, which is
   closer to what a new user's photos would be?

## 2. Data

| Set | Source | Images | Used for |
|---|---|---|---|
| PlantDoc field-train / val / test | PlantDoc, Phase 6 split (unchanged) | 1,278 / 381 / 898 | train / selection / report |
| FieldPlant-train / val / test | FieldPlant (Moupojou et al., 2023), CC BY 4.0 | 1,159 / 349 / 812 leaf crops | train / selection / report |
| CD&S | CD&S (Ahmad et al., 2021), corn Gray leaf spot and Northern Leaf Blight, `Dataset_Original` | 1,020 | **report only** (independent source) |
| Lab check | leak-free clean test, 20 per class (as Phase 6) | 746 | forgetting |
| Lab replay | clean train, 50 per class (as the deployed Phase 6 model) | 1,900 | mixed into training |

- **FieldPlant** (`python -m leafdx.data.extra_field fieldplant`): 5,156 photos from plantations
  in Cameroon, with leaves boxed under the supervision of plant pathologists. Its class names
  follow PlantDoc's, and the same mapping rules are applied
  (`ml/resources/fieldplant_to_plantvillage.csv`). Seven classes map to the model's 38: corn
  Northern Leaf Blight, Gray leaf spot, Common rust and healthy; tomato healthy, Yellow Leaf
  Curl Virus and mosaic virus. Cassava, other corn disorders and the ambiguous tomato classes
  ("Brown Spots", "blight leaf") are not used. Each leaf box becomes one image (5% margin).
  Crops from one photo, and from near-duplicate photos (D4 pHash ≤ 6 bits), stay in one split;
  split 50/15/35, stratified by class, seed 42. Northern Leaf Blight is 71% of the crops.
- **CD&S** (`python -m leafdx.data.extra_field cds`): handheld field photos of corn from the
  United States. Its OSF project states no licence, so it is **used only for testing**, never for
  training, and its images are not redistributed. Only its two classes that exist in the
  model are used (Northern Leaf Spot has no counterpart).
- **Plant Pathology 2020** (apple) is not used: its competition rules have not been accepted,
  and its terms could not be checked.

## 3. Starting models

Seed-42 Phase 3 models (`ml/models/_export/p3_*/inference.keras`), as in Phase 6:
**MobileNetV1** (99.06% lab check) and **EfficientNetB0** (99.60% lab check).

## 4. Methods

Recipe from Phase 6, unchanged unless stated:
- fine-tune the whole network, Adam 3e-5, batch 32;
- at most 15 epochs, early stopping on the loss of the pooled validation set (PlantDoc-val +
  FieldPlant-val) with patience 3, best weights restored;
- batch normalisation in inference mode;
- random resized crop, flips and colour jitter;
- 50 lab replay images per class;
- 3 seeds (0, 1, 2).

| Id | Field training data |
|---|---|
| **P** | PlantDoc field-train (as Phase 6) |
| **PF** | PlantDoc field-train + FieldPlant-train, with FieldPlant capped at **100 crops per class** (so Northern Leaf Blight does not dominate; 390 crops) |
| **PF-long** | as PF, at most 30 epochs with patience 5. Run **only if** in PF the best epoch is ≥ 13 for at least two of the three seeds of that model (i.e. training was still improving) |

Each method is run for both models. Every model is evaluated with and without test-time
augmentation (4 views, as the app). The deployed Phase 6 model is also evaluated on the new
test sets, as the "before" reference.

## 5. Metrics and decision rule

- **Selection score (field-val):** the mean of PlantDoc-val accuracy and FieldPlant-val
  balanced accuracy (mean per-class recall). Balanced accuracy is used for FieldPlant because
  of its class imbalance.
- **Reported (test):**
  - PlantDoc field-test accuracy (primary; comparable with Phase 6);
  - FieldPlant-test accuracy and balanced accuracy;
  - CD&S accuracy (38-way prediction, as everywhere) and corn accuracy;
  - macro-F1 and ECE;
  - lab check accuracy;
  - for the deployed model, the full leak-free lab test (11,599).
- **Deployment rule.** Among methods with lab check accuracy within 2 pp of their own starting
  model, take the best on the selection score (TTA counted as part of the method). Deploy it
  only if:
  - its TF.js float16 model is at most 10 MB;
  - it passes the browser checks: same top-1 as Keras on ≥ 98% of verification images, with
    the same check for TTA and Grad-CAM;
  - it beats the deployed Phase 6 model on the selection score.

  Within the chosen method, the seed with the best selection score is deployed, and the
  method's mean over seeds is reported next to it (as in Phase 6). Test sets are not used for
  any choice.

## 6. Results

*Pending.*

## 7. Deviations from this protocol

1. **Overfitting checks (added 2026-11-19, before any Phase 7 fine-tuning run had finished;
   only the reference evaluations of existing models had been seen).**
   `python -m leafdx.field.phase7 diagnose` adds, for every fine-tuned run:
   - **train–validation gap:** accuracy on the run's own field training images (PlantDoc
     field-train; FieldPlant-train balanced accuracy for PF), without augmentation, minus the
     same metric on validation;
   - **best epoch** and training/validation loss curves (`figures/phase7/loss_curves.png`);
   - **95% bootstrap intervals** (1,000 resamples) for every test score.

   The deployment rule (§5) is tightened in two ways:
   - **No lucky seed.** The chosen method must beat the deployed Phase 6 model on its *mean*
     selection score over three seeds, not only on its best seed.
   - **Simpler model on a tie.** If two eligible methods differ in mean selection score by less
     than the larger of their seed standard deviations, the one with the smaller
     train–validation gap is preferred (on a further tie, the smaller model).

   Test sets are still used for no choice.
