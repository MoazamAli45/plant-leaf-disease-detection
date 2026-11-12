# Dataset card

> **Status:** Phase 1. Every `⟨PLACEHOLDER⟩` below is filled from
> `docs/figures/data_audit/summary.json` once `make prepare` has run on the Kaggle data.
> No number in this card is typed in by hand.

## 1. Datasets used

| Role | Dataset | Version pinned | Images | License |
|---|---|---|---|---|
| Train / val / test (in-distribution) | New Plant Diseases Dataset (NPD) [`npd_kaggle`] | Kaggle version 2 (last updated 2018-11-18) | ⟨PLACEHOLDER: `npd.n_images`⟩ | "Data files © Original Authors": upstream PlantVillage terms apply |
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
PlantVillage name and add a suffix per transform (`_90deg`, `_180deg`, `_270deg`, `_flipLR`,
`_flipTB`, `_new30degFlipLR`, `_new30degFlipTB`, `_newGRR`). If the 80/20 split was drawn
*after* augmentation, rotated or mirrored copies of one photo can sit on both sides of the split.
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

| Measure | Value |
|---|---|
| val images whose source photo has a copy in train | ⟨PLACEHOLDER: `paper_split_audit.frac_source_in_train`⟩ |
| val images whose physical leaf appears in train | ⟨PLACEHOLDER: `paper_split_audit.frac_leaf_in_train`⟩ |
| val images with a byte-identical file in train | ⟨PLACEHOLDER: `paper_split_audit.frac_exact_duplicate_in_train`⟩ |
| any of the above | ⟨PLACEHOLDER: `paper_split_audit.frac_source_or_leaf_or_exact`⟩ |
| pHash near-duplicate in train, by threshold | ⟨PLACEHOLDER: `figures/data_audit/paper_split_leakage_thresholds.md`⟩ |

## 5. Clean protocol: grouped, stratified 70/15/15 split

1. Build a graph over all NPD images (train + valid merged). Images are linked when they share
   a source photo, a physical leaf or a SHA-256, or when their D4 pHash distance is
   ≤ `phash.group_threshold` (default 4 bits). The threshold must be justified by the
   "hash-only pairs" figure in the EDA notebook.
2. Connected components are **groups**. No group is ever split between train, val and test.
3. Groups that contain more than one label (the same picture filed under different classes)
   are dropped as label conflicts. Byte-identical copies are collapsed to one file.
4. Within each class, groups are assigned largest first (seeded random order among equal
   sizes) to the split furthest below its 70/15/15 target. Seed: 42 (`configs/data.yaml`).
5. The pipeline **asserts** that val and test share no source photo, leaf, file or
   near-duplicate (≤ threshold) with train, and fails otherwise.

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
same class, one copy is kept. Kept / excluded:
⟨PLACEHOLDER: `plantdoc.excluded`⟩. Phase 3 reports field accuracy on all mapped classes and on
the `exact`-only subset.

## 7. Known issues and limitations

- **Lab capture bias.** PlantVillage photos were taken on uniform backgrounds under controlled
  light. A classifier trained on only 8 background pixels reaches 49% accuracy on it, against
  2.6% for chance [`noyan2022uncovering`]. In-distribution accuracy therefore overstates field
  performance. The EDA notebook compares background brightness between NPD and PlantDoc.
- **Leaf ids are incomplete.** Upstream `leaf-map.json` covers only part of PlantVillage
  (⟨PLACEHOLDER: `npd.n_with_leaf_id`⟩ of NPD's images resolve to a leaf). Photos of the same
  leaf without a leaf id are linked only if pHash catches them.
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
