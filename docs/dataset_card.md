# Dataset card

> **Status:** final (Phase 5). The numbers below come from the run of 2026-11-13
> (`docs/figures/data_audit/summary.json`; manifests verified by `ml/manifests/SHA256SUMS`).
> Re-running `make prepare` reproduces them byte for byte.
>
> **Correction (Phase 3).** The Phase 1 file-name parser missed NPD's `_newPixel25`
> augmentation (2,399 images). With it, **90.31%** of the paper split's validation images are
> linked to training (§4 below reports the original 87.5%), and 11.94% of the clean split v1
> test images are linked to training, concentrated in five classes. Results are therefore
> reported on the 11,599-image leak-free test subset. Details and the re-audit:
> [`experiments.md` §2](experiments.md#2-a-correction-to-the-clean-split),
> `figures/data_audit/recheck_newpixel.json`.

## 1. Datasets used

| Role | Dataset | Version pinned | Images | License |
|---|---|---|---|---|
| Train / val / test (in-distribution) | New Plant Diseases Dataset (NPD) [`npd_kaggle`] | Kaggle version 2 (last updated 2018-11-18) | 87,867 (70,295 train + 17,572 valid; the archive stores every file twice in identical trees, and 33 unlabelled `test/` images are ignored) | "Data files © Original Authors": upstream PlantVillage terms apply |
| Upstream of NPD | PlantVillage [`hughes2015open`, `mohanty2016using`] | git `7f7ecc7` (leaf map only) | not downloaded | CC BY-SA 3.0 (see note) |
| Field test set (out-of-distribution) | PlantDoc classification set [`singh2020plantdoc`] | git `5467f60` | 2,578 files extracted (from `ml/data/raw/PROVENANCE.json`) | CC BY 4.0 |

**License notes.**
- **PlantVillage** licensing is reported inconsistently. The maintainers' own Hugging Face card
  (`mohanty/PlantVillage`) and loader declare **CC BY-SA 3.0**. Third-party mirrors (Meta-Album,
  OpenML) list CC0 1.0, and one Kaggle mirror (`abdallahalidev/plantvillage-dataset`) relabels
  it CC BY-NC-SA 4.0. We follow the maintainers' statement. CC BY-SA 3.0 requires attribution,
  and *ShareAlike* applies to adapted versions of the dataset itself (for example a
  redistributed copy or a re-augmented release). Whether trained model weights count as an
  adaptation is legally unsettled. We attribute the dataset and do not redistribute images.
- **NPD** is a derivative of PlantVillage and states no license of its own ("© Original
  Authors"), so it inherits PlantVillage's terms.
- **PlantDoc** is CC BY 4.0 (confirmed with the GitHub license API for
  `pratikkayal/PlantDoc-Dataset`).
- This repository never redistributes images. It ships download scripts and CSV manifests
  (file paths, labels and hashes) only.
  The one exception is the small sample grids in `docs/figures/eda/`. They show a few dataset
  images under the licenses above, with attribution to the datasets cited here.

## 2. Provenance and download

```bash
cd ml
make setup
make data          # Kaggle token required for NPD; `make data-public` fetches the rest
make prepare       # audit + manifests
make eda           # figures
```

`ml/src/leafdx/data/sources.py` pins every source. A download fails if the Kaggle version is
not the expected one. `ml/data/raw/PROVENANCE.json` records the archive SHA-256, the Kaggle
metadata (version, license field) and the retrieval time.

**Extraction caveat (found during this work).** The PlantDoc archive contains 6 pairs of files
whose names differ only by case (for example `CAR1.jpg` and `car1.jpg`) and whose content
differs. On case-insensitive file systems (the macOS and Windows defaults) a plain unzip
silently loses one file of each pair. The extractor renames the second file to
`<name>__case2.<ext>` and lists every rename in `PROVENANCE.json`.

## 3. How NPD was built, and why it leaks

NPD's Kaggle description says it was "recreated using offline augmentation from the original
dataset" and then "divided into 80/20 ratio of training and validation set". File names keep the
PlantVillage name and add a suffix per transform. The 87,867 images come from **48,239 source
photos**:

| Suffix | Images | Suffix | Images |
|---|---|---|---|
| (none: original) | 48,239 | `_new30degFlipLR` | 5,042 |
| `_flipLR` | 8,681 | `_new30degFlipTB`, `_newGRR` | 427 each |
| `_180deg` | 8,668 | `_new200degFlipLR/TB`, `_new90degFlipLR/TB`, `_newGGR` | 152 each |
| `_flipTB` | 7,275 | `_90deg`, `_270deg` | 4,174 each |

If the 80/20 split was drawn *after* augmentation, rotated or mirrored copies of one photo can
sit on both sides of the split.
PlantVillage also often has several photos of one physical leaf, which upstream records in
`leaf_grouping/leaf-map.json`. Both effects let a validation image be "seen" during training.

## 4. Audit protocol

For each validation image of the paper's split, we test whether the training split contains:

| Link | How it is detected |
|---|---|
| same source photo | file name with the augmentation suffix stripped (`leafdx.data.filenames`) |
| same physical leaf | upstream `leaf-map.json`; the leaf id is accepted only when its class matches the image's label (stricter than the official loader, because photo identifiers repeat across crops) |
| byte-identical file | SHA-256 |
| perceptual near-duplicate | 64-bit DCT pHash [`zauner2010phash`], made **D4-invariant**: 8 hashes per image (4 rotations × mirror); distance = minimum Hamming distance over transforms. Searched exhaustively *within* each class; across classes only exact D4 matches are detected |

pHash is also *calibrated* against the known (original, augmented copy) pairs
(`phash_recall_by_aug_type.csv`). Recall per augmentation type shows which leaks pHash alone
would miss. Arbitrary-angle rotations, for example, change the hash.

### Results on the paper's 80/20 split

All 17,572 validation images were audited.

| Measure | Images | Share |
|---|---|---|
| source photo has an augmented copy in train | 10,974 | 62.5% |
| physical leaf also photographed in train | 11,988 | 68.2% (98.5% of the 12,166 val images that have a leaf id) |
| byte-identical file in train | 6 | 0.03% |
| **any of the above** | **15,383** | **87.5%** |

**87.5% is a lower bound** (90.31% after the Phase 3 correction above). Leaf ids exist for only 69.2% of images, so photos of the same leaf
without an id are not counted.

**pHash as a detector.** Recall on the known (original, copy) pairs
(`phash_recall_by_aug_type.csv`):

- **90/180/270-degree rotations and flips:** ≥ 99.7% at ≤ 4 bits.
- **The `new*` 30- and 200-degree rotations:** 0% at any threshold tested. Their median
  distance is 22–24 bits.

So pHash alone would miss about 5,800 copies that file names catch. In the other direction,
pHash at ≥ 2 bits also matches *different* leaves on the same lab background: even in the clean
split, 0.8% of test images have a train "near-duplicate" at 2 bits and 71% at 12 bits.
Near-duplicate rates on the paper split (`paper_split_leakage_thresholds.md`) therefore mix
real leaks with look-alikes, and the file-name and leaf-id numbers above are the defensible
measure.

| pHash threshold (bits) | paper val with a train near-duplicate | of which not explained by file name or leaf id |
|---|---|---|
| 0 | 52.3% | 0.03% |
| 4 | 58.6% | 0.9% |
| 8 | 70.9% | 3.1% |
| 12 | 90.9% | 7.7% |

Figures: `docs/figures/eda/leakage_curve.png`, `augmented_variants.png`.

## 5. Clean protocol: grouped, stratified 70/15/15 split

1. Build a graph over all NPD images (train + valid merged). Images are linked when they share
   a source photo, a physical leaf or a SHA-256, or when their D4 pHash distance is
   ≤ `phash.group_threshold` = **0 bits** (an exact match under rotation and flip). The
   threshold was chosen from the first full run:
   - by eye (`hash_only_pairs.png`), the 85 pairs at 0 bits not explained by file name or leaf
     id are true duplicates; pairs at 2 bits are mixed; pairs at ≥ 4 bits are different
     leaves;
   - grouping is single-linkage, so a loose threshold chains look-alikes together. At ≤ 0 bits
     the largest group has 110 images (4.9% of its class). At ≤ 2 bits it has 594 (26.7%). At
     ≤ 4 bits it has 1,870, which is 84% of *Raspberry healthy* and would leave that class
     with almost no val or test images.
2. Connected components are **groups**. No group is ever split between train, val and test.
3. Groups that contain more than one label (the same picture filed under different classes)
   are dropped as label conflicts. Byte-identical copies are collapsed to one file.
4. Within each class, groups are assigned largest first (seeded random order among equal
   sizes) to the split furthest below its 70/15/15 target. Seed: 42 (`configs/data.yaml`).
5. The pipeline **asserts** that val and test share no source photo, leaf, file or
   near-duplicate (≤ threshold) with train, and fails otherwise.

**Result.** The split has 22,133 groups (median size 2, largest 110). It holds **61,485 train
/ 13,185 val / 13,171 test** images. 26 byte-identical copies were removed, and no group
contained two labels. Every class is within 0.25 percentage points of 70/15/15, and every class
has at least 308 val and 308 test images. Val and test share no source photo, leaf or file with
train, and have no train image at 0 bits.

Manifests: `ml/manifests/clean_split/{train,val,test}.csv` (columns: `path, label, group_id,
source_id, leaf_id, aug_type, sha256`). `ml/manifests/SHA256SUMS` makes the split citable and
verifiable. The paper's original split is kept for the Phase 2 reproduction in
`ml/manifests/paper_split/`.

The val and test sets still contain offline-augmented copies (always inside the same group as
their source). Phase 3 reports test metrics on all test images and on originals only
(`aug_type == "original"`).

## 6. PlantDoc field test set and class mapping

PlantDoc is used **only for evaluation**, with its own train and test folders combined. Of its
28 classes, all 28 map to a PlantVillage class. 10 PlantVillage classes have no PlantDoc
counterpart. The mapping lives in `ml/resources/plantdoc_to_plantvillage.csv`, and a test checks
it against the real folder names.

| Match type | Meaning | Classes |
|---|---|---|
| `exact` | same crop and same disease name | 13 |
| `healthy_assumed` | PlantDoc's `<crop> leaf` folders are its healthy classes | 10 |
| `approximate` | plausible but not certain; see the rationale column | 5 |

Approximate mappings:

| PlantDoc class | PlantVillage class | Why approximate |
|---|---|---|
| Apple rust leaf | Apple___Cedar_apple_rust | rust species not named; cedar apple rust is PlantVillage's only apple rust |
| Bell_pepper leaf spot | Pepper,_bell___Bacterial_spot | "leaf spot" is not pathogen-specific |
| Corn leaf blight | Corn_(maize)___Northern_Leaf_Blight | northern and southern blight not distinguished |
| Corn rust leaf | Corn_(maize)___Common_rust_ | common and southern rust not distinguished |
| Tomato leaf yellow virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus | taken to mean TYLCV; not confirmed by the PlantDoc authors |

PlantVillage classes with no PlantDoc counterpart: Apple black rot, Cherry powdery mildew, Corn
healthy, Grape esca, Grape leaf blight, Orange Huanglongbing, Peach bacterial spot, Potato
healthy, Strawberry leaf scorch, Tomato target spot.

**Overlap removal.** PlantDoc was collected from the web, so it may contain PlantVillage images.
PlantDoc images within `plantdoc.overlap_threshold` bits (D4 pHash, all classes) of any NPD
image are excluded. Byte-identical files filed under **different** PlantDoc classes are
excluded entirely, because their true label cannot be known. For identical copies under the
same class, one copy is kept. The overlap threshold is 0 bits. At 4 bits, 8 PlantDoc images
matched NPD images, and by eye all 8 were unrelated leaves (mostly stock photos on white
paired with PlantVillage leaves of other crops).

**Result:** 2,557 of 2,578 images kept, covering all 28 classes. 18 were removed as label
conflicts (9 pairs), 3 as same-label duplicates and 0 as NPD copies. Phase 3 reports field
accuracy on all mapped classes and on the `exact`-only subset.

## 7. Known issues and limitations

- **Lab capture bias.** PlantVillage photos were taken on uniform backgrounds under controlled
  light. A classifier trained on only 8 background pixels reaches 49% accuracy on it, against
  2.6% for chance [`noyan2022uncovering`]. In-distribution accuracy therefore overstates field
  performance. The EDA notebook compares background brightness between NPD and PlantDoc.
- **Leaf ids are incomplete.** Upstream `leaf-map.json` covers only part of PlantVillage
  (60,809 of NPD's 87,867 images, 69.2%, resolve to one of 7,378 leaves). Photos of the same
  leaf without a leaf id are linked only if pHash matches them exactly, so some same-leaf
  pairs may remain across the clean split. Residual look-alike rates are reported in
  `clean_split_test_leakage.json`.
- **Rotation artefacts.** The `new*` rotations fill their corners with black. This is a
  synthetic cue the original photos do not have.
- **PlantDoc duplicates with conflicting labels.** Indexing the 2,578 PlantDoc files found 12
  byte-identical pairs. In 9 of them the two copies carry different labels: 5 pairs are filed
  under both *Potato leaf early blight* and *late blight*, 3 under both *Corn Gray leaf spot*
  and *Corn leaf blight*, and 1 under both *Tomato Septoria leaf spot* and *Tomato leaf
  bacterial spot*. 11 of the 12 pairs span PlantDoc's own train/test split, so they also leak
  in PlantDoc's original benchmark. (Counts from `ml/data/cache/index_plantdoc.csv`; the
  pipeline's `plantdoc_classes.csv` reports the exclusions.)
- **PlantDoc label noise.** Observed while building this pipeline:
  - a file named `potato-blight-phytophora-infestans…` sits in *Potato leaf early blight*,
    although *Phytophthora infestans* causes **late** blight;
  - several images carry stock-photo watermarks;
  - at least one image in *Tomato two spotted spider mites leaf* is a captioned collage of
    several diseases.

  These are left in place, because cleaning a test set by hand biases it. They are reported as
  a limitation instead.
- **Class imbalance.** NPD's offline augmentation was meant to balance classes, so the balance
  in NPD differs from the original PlantVillage. See `docs/figures/eda/class_counts.png`.

## 8. Intended use

Research on plant leaf disease classification and on evaluation methodology (leakage,
generalisation to field images). Not validated for agronomic decisions.

## References

See `docs/references.bib` (keys in brackets above).
