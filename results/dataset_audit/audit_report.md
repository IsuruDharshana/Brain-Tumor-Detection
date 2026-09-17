# Dataset Audit Report

_Generated: 2026-09-17T16:20:59Z - Phase 2 (read-only audit; no image was modified, resized or deleted)._

## Dataset
- Name: Brain Tumor MRI Dataset
- Kaggle identifier: `masoudnickparvar/brain-tumor-mri-dataset` (public dataset, CC BY 4.0)
- Source: downloaded with the Kaggle CLI; not collected by this project
- Local location: `data/raw` (excluded from Git)
- Audit script: `python -m src.data.audit_dataset`

## Dataset structure
- Detected splits: Training, Testing
- Detected classes: glioma, meningioma, notumor, pituitary
- Structural anomalies: none detected

## Total images
- Total image files: **7,200**
- Valid images: **7,200**
- Invalid / unreadable images: **0**
- Non-image files inside class folders: **0**
- Total dataset size: 160.07 MB

## Class distribution
| split | class | count | % of split | % of dataset |
|---|---|---|---|---|
| Training | glioma | 1,400 | 25.00% | 19.44% |
| Training | meningioma | 1,400 | 25.00% | 19.44% |
| Training | notumor | 1,400 | 25.00% | 19.44% |
| Training | pituitary | 1,400 | 25.00% | 19.44% |
| Testing | glioma | 400 | 25.00% | 5.56% |
| Testing | meningioma | 400 | 25.00% | 5.56% |
| Testing | notumor | 400 | 25.00% | 5.56% |
| Testing | pituitary | 400 | 25.00% | 5.56% |

- Largest class (dataset-wide): glioma (1,800)
- Smallest class (dataset-wide): glioma (1,800)
- Largest-to-smallest ratio: 1.0

## Train/Test distribution
| split | count | % of dataset |
|---|---|---|
| Training | 5,600 | 77.78% |
| Testing | 1,600 | 22.22% |

## File formats
| image extension | count |
|---|---|
| .jpg | 7,200 |

Format/extension mismatches (file content vs filename): **4** - see `dataset_summary.json` for examples.

## Image dimensions
- Width range: 150 - 1375 px
- Height range: 167 - 1446 px
- Unique dimension combinations: **447**

| width | height | count | % of valid images |
|---|---|---|---|
| 512 | 512 | 5,014 | 69.64% |
| 225 | 225 | 338 | 4.69% |
| 630 | 630 | 90 | 1.25% |
| 201 | 251 | 57 | 0.79% |
| 228 | 221 | 51 | 0.71% |
| 232 | 217 | 50 | 0.69% |
| 442 | 442 | 48 | 0.67% |
| 236 | 236 | 48 | 0.67% |
| 150 | 198 | 44 | 0.61% |
| 200 | 252 | 43 | 0.60% |

No resizing was performed during this audit (a target size decision belongs to Phase 3).

## Image modes/channels
| PIL mode | count |
|---|---|
| RGB | 4,129 |
| L | 3,067 |
| RGBA | 3 |
| P | 1 |

- Grayscale (`L`): 3,067 | RGB: 4,129 | RGBA: 3 | unusual modes: {'P': 1}

Note: an MRI can visually look grayscale while being stored as RGB; the mode above reflects the stored pixel format, not the visual impression.

## Corrupt files
No corrupt or unreadable images were detected.

## Exact duplicates
- Duplicate groups: **153**
- Files inside duplicate groups: **340** (4.72% of valid images)
- Groups fully inside one split: {'Training': 138, 'Testing': 15}
- Groups crossing splits: **0**
- Groups with conflicting class labels for identical content: **0**

Full listing: `exact_duplicates.csv` (and `cross_split_duplicates.csv` for leakage-relevant groups).

## Cross-split duplicates
No exact-duplicate group overlaps between Training and Testing was detected by SHA-256 hashing.

## Perceptual near-duplicates (heuristic)
- Method: pHash (64-bit), Hamming distance <= 8
- Images hashed: 7,200 (failures: 0)
- Near-duplicate group(s): **809** covering 6,132 files (largest group: 2,789)
- Groups crossing splits: 454; conflicting-label groups: 45

Heuristic similarity only - not proof of duplication. Never delete files based on this section alone. Groups are connected components at the threshold, so very large group(s) can reflect transitive similarity chaining rather than one duplicate cluster. See `perceptual_duplicates.csv` (reported separately from exact SHA-256 duplicates).

## Potential leakage concerns
Duplicates exist inside splits (153 group(s)) but no exact-content overlap between Training and Testing was detected.

## Observations
- 4 image(s) whose real format does not match their extension.

## Decision required before Phase 3
- Decide how to handle cross-split exact duplicates (drop duplicates, relabel, or rebuild the train/test split).
- Decide class-imbalance handling (report only - see distribution above).
- Decide the input image size (224x224 is the current plan) and colour handling (grayscale vs RGB) for preprocessing.
- Decide the policy for any corrupt / unusual files listed above.
- Decide whether perceptual findings need manual review.

