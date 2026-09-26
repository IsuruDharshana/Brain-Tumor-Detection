# Brain Tumor MRI Classification

A university machine-learning project that trains a Convolutional Neural
Network (CNN) to classify brain MRI scans into tumour categories.

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment setup | ✅ Complete |
| 2 | Dataset acquisition and audit | ✅ Complete |
| 3 | Preprocessing and tf.data pipeline | ✅ Complete |
| 4 | Baseline custom CNN | ✅ Complete |
| 5 | EfficientNetB0 transfer learning | ✅ Complete |
| 6 | Final evaluation + model comparison + Grad-CAM | ✅ Complete |
| 7 | Streamlit application | ⏳ Pending |

---

## Project Structure

```
brain-tumor-classification/
├── app/                  # Streamlit web UI (Phase 3+)
├── data/
│   ├── raw/              # Original, unmodified dataset (not committed)
│   └── processed/        # Split manifests (small committed CSVs; no image copies)
├── models/               # Saved model weights (not committed)
├── notebooks/            # Exploratory data-analysis notebooks
├── results/              # Plots, metrics, confusion matrices
├── src/
│   ├── data/             # Dataset loading & pre-processing
│   ├── models/           # Model architectures
│   ├── evaluation/       # Metrics, visualisation helpers
│   └── utils/            # Shared utilities
├── tests/                # Unit and integration tests
├── .gitignore
├── README.md             ← you are here
├── requirements.txt
└── environment_check.py
```

---

## Setup

### 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12.x |
| Git | any recent version |

### 2. Clone the repository

```powershell
git clone <repo-url>
cd brain-tumor-classification
```

### 3. Create the virtual environment

```powershell
py -3.12 -m venv .venv
```

### 4. Activate the virtual environment (Windows)

```powershell
.venv\Scripts\Activate.ps1
```

> **Tip:** If PowerShell blocks script execution, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 5. Install dependencies

```powershell
pip install -r requirements.txt
```

### 6. Verify the environment

```powershell
python environment_check.py
```

Expected output on this laptop:

- TensorFlow imported successfully ✓
- **0 GPU devices** — normal; GPU is on the training workstation

---

## GPU Training Note

Model training (`src/models/`) will run on the **RTX 3060** workstation.
The workstation environment will use the same `requirements.txt` plus the
appropriate CUDA-enabled TensorFlow build.

---

## Dataset

The project uses the public **Brain Tumor MRI Dataset**:

| | |
|---|---|
| Kaggle identifier | `masoudnickparvar/brain-tumor-mri-dataset` |
| Classes | `glioma`, `meningioma`, `notumor`, `pituitary` |
| Splits | `Training/` and `Testing/` |
| License | CC BY 4.0 (as listed on Kaggle) |

The dataset was **not collected by this project** and does not represent all
real-world MRI populations; it is used for educational/research purposes
only. Raw data is intentionally **excluded from Git** — every developer must
download it themselves, and the audit scripts reproduce the validation
process.

### Downloading the dataset (reproducible)

A Kaggle account (and, depending on your account settings, an API token) may
be required. If the CLI asks for authentication, run `kaggle auth login` or
place a token in `~/.kaggle/access_token`. **Never commit Kaggle
credentials.**

```powershell
python -m pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/raw
```

Extract the archive into `data/raw` so the layout becomes
`data/raw/Training/<class>/*.jpg` and `data/raw/Testing/<class>/*.jpg`:

```powershell
python -m zipfile -e data/raw/brain-tumor-mri-dataset.zip data/raw/
Remove-Item data/raw/brain-tumor-mri-dataset.zip  # only after a successful extraction
```

### Dataset audit (Phase 2)

The audit is **read-only** — it never modifies, renames or deletes anything
inside `data/raw`. Run it from the repository root:

```powershell
python -m src.data.audit_dataset
```

It validates the directory structure, image integrity, dimensions, colour
modes, exact SHA-256 duplicates, cross-split (Train/Test) leakage and
optionally perceptual (pHash) near-duplicates, then writes a report and
machine-readable summaries to `results/dataset_audit/`
(`audit_report.md`, `dataset_summary.json`, CSV tables and PNG charts).

Notes:

- `--data-dir` / `--output-dir` override the default paths
  (`data/raw`, `results/dataset_audit`).
- `--skip-perceptual` skips the optional pHash near-duplicate stage.
- The large per-file table `image_metadata.csv` is generated locally and is
  intentionally kept out of Git.

---

## Preprocessing and data pipeline (Phase 3)

Phase 3 converts the audited raw dataset into a reproducible training /
validation / testing setup. **No model training happens in this phase.**

Design decisions:

- `data/raw` stays **immutable** — nothing there is modified, renamed or
  deleted; all cleaning is expressed through manifests only.
- **Exact duplicates are removed from the Training pool logically**: in every
  SHA-256 duplicate group the lexicographically first path is kept and the
  other members are excluded from the processed pool (they stay on disk).
  Only exact SHA-256 duplicates are used — perceptual (pHash) similarity is
  never used to remove images.
- **Validation is created from the deduplicated Training pool only** with a
  deterministic, stratified 80/20 train/validation split (seed **42**,
  per-class shuffle). Duplicates can therefore never be split between
  train and validation.
- The **original Kaggle Testing split stays fully isolated** and unchanged;
  its internal duplicates are kept for the official evaluation and listed
  separately in `results/preprocessing/test_duplicates.csv` for a later
  deduplicated-test sensitivity analysis.
- Every image is decoded (JPEG or PNG content), converted to **RGB**,
  resized to **224×224**, kept as **float32** and normalised to **[0, 1]**
  (division by 255.0).
- **Mild augmentation is applied to the training split only** (rotation ±8°,
  zoom ≤8%, translation ≤5%, mild contrast; no flips, shear or crops — MRI
  left/right orientation can carry meaning). Validation/test pipelines are
  deterministic and unaugmented.
- Preprocessing happens **on the fly** in the tf.data pipeline — no resized
  image copies are written to disk.

| Artefact | Location |
|---|---|
| Manifests (portable relative paths) | `data/processed/{train,val,test}_manifest.csv` |
| Deduplication report | `results/preprocessing/training_deduplication.csv` |
| Test duplicate metadata (Phase 6 aid) | `results/preprocessing/test_duplicates.csv` |
| Split summary (seed, counts, mapping) | `results/preprocessing/split_summary.json` |
| Class distribution | `results/preprocessing/class_distribution.csv` |
| Preprocessing sanity image | `results/preprocessing/preprocessing_samples.png` |

Label mapping (fixed, independent of filesystem order):
`glioma=0, meningioma=1, notumor=2, pituitary=3` — integer labels, ready for
`SparseCategoricalCrossentropy` in Phase 4.

### Commands

```powershell
python -m src.data.create_splits          # regenerate manifests + summaries (deterministic)
python -m src.data.verify_pipeline        # leakage checks + one real batch per pipeline
python -m src.data.preprocessing_samples  # optional visual sanity check (PNG)
```

---

## Running Tests

```powershell
python -m pytest
```

The suite covers the Phase 2 dataset audit and the Phase 3 preprocessing /
split / tf.data pipeline logic. It runs on small synthetic images and does
not require the full Kaggle dataset.

---

## Baseline CNN (Phase 4)

Phase 4 establishes a reproducible baseline custom CNN **trained entirely from
scratch** (no pretrained weights). Its sole purpose is to provide a fair
scientific reference point for the EfficientNetB0 transfer-learning model built
in Phase 5 and evaluated head-to-head in Phase 6.

### Purpose

A credible baseline that is deliberately simple:

- Confirms the tf.data pipeline and training infrastructure work end-to-end.
- Gives a meaningful lower-bound accuracy against which transfer learning will
  be compared.
- Demonstrates what a straightforward CNN can achieve on 4,344 training images.

### Architecture

```
Input (224 × 224 × 3)
│
├─ Block 1: Conv2D(32, 3×3, same, relu) → BatchNorm → MaxPool2D → (112, 112, 32)
├─ Block 2: Conv2D(64, 3×3, same, relu) → BatchNorm → MaxPool2D →  (56,  56, 64)
├─ Block 3: Conv2D(128,3×3, same, relu) → BatchNorm → MaxPool2D →  (28,  28,128)
├─ Block 4: Conv2D(256, 3×3, same, relu) → BatchNorm            →  (28,  28,256)
│
├─ GlobalAveragePooling2D → (256,)
├─ Dense(128, relu)
├─ Dropout(0.4)
└─ Dense(4, softmax, float32) → (4,)
```

No pretrained weights. No EfficientNet, MobileNet, ResNet, VGG, DenseNet or
any other transfer-learning backbone.

### Training configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss | SparseCategoricalCrossentropy |
| Metrics | accuracy |
| Batch size | 32 |
| Maximum epochs | 30 |
| Seed | 42 |

### Callbacks

| Callback | Setting |
|----------|---------|
| ModelCheckpoint | monitor `val_loss`, save best only → `models/baseline_cnn.keras` |
| EarlyStopping | monitor `val_loss`, patience 5, `restore_best_weights=True` |
| ReduceLROnPlateau | monitor `val_loss`, factor 0.5, patience 2, min LR 1e-6 |

### Training

Training runs on the **NVIDIA RTX 3060 (12 GB)** GPU workstation under
WSL2/Ubuntu. The CPU laptop is used for development only.

```bash
# From the repository root, inside the virtual environment
python -m src.models.train_baseline
```

### Outputs

| Artefact | Location |
|----------|----------|
| Best model checkpoint | `models/baseline_cnn.keras` (**not committed** — large binary) |
| Model summary | `results/baseline_cnn/model_summary.txt` |
| Per-epoch metrics CSV | `results/baseline_cnn/training_history.csv` |
| Training curves PNG | `results/baseline_cnn/training_curves.png` |
| Results summary | `results/baseline_cnn/training_summary.json` |
| Experiment config | `results/baseline_cnn/run_config.json` |

### Test-set evaluation

**The official Testing split (1,600 images) is NOT evaluated in Phase 4.**
It remains completely untouched and is reserved for the final head-to-head
comparison in Phase 6 (baseline CNN vs. EfficientNetB0).

---

## EfficientNetB0 Transfer Learning (Phase 5)

Phase 5 implements **ImageNet transfer learning** using EfficientNetB0 as a
feature extractor. The model is trained in two stages on the same data splits
as Phase 4, using validation metrics only for model selection. The test set
remains reserved for Phase 6.

### Preprocessing strategy

The Phase 3 tf.data pipeline delivers images as **float32 in [0, 1]**.
EfficientNetB0 in TF/Keras 2.x includes an internal `Rescaling(1/255)` layer
and expects **[0, 255]** float32 inputs.

To avoid double normalisation while keeping the shared data pipeline
unchanged, the model includes an explicit **model-side adapter layer**:

```
Input [0, 1]  (Phase 3 pipeline output)
    │
    ▼  Rescaling(scale=255.0)  ← adapter inside the model
    │
    ▼  [0, 255] float32
    EfficientNetB0 (internal Rescaling(1/255) → ImageNet feature space)
```

`tf.keras.applications.efficientnet.preprocess_input` is **not** called —
it is a pass-through in TF 2.x and adds no value here.

### Model architecture

```
Input (224 × 224 × 3) — float32, [0, 1]
    │
    ▼ Rescaling(255.0)               adapter: [0,1] → [0,255]
    ▼ EfficientNetB0(include_top=F)  ~4 M params; internal Rescaling(1/255)
    ▼ GlobalAveragePooling2D         → (1280,)
    ▼ Dropout(0.3)
    ▼ Dense(128, relu)
    ▼ Dropout(0.3)
    └ Dense(4, softmax, float32)     → (4,)
```

No pretrained weights are used for the classifier head. No Flatten layer.

### Transfer learning strategy

**Stage 1 — Frozen backbone (head training)**

| Setting | Value |
|---------|-------|
| EfficientNetB0 | fully frozen |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Max epochs | 15 (EarlyStopping patience 4) |
| Batch size | 32 |

**Stage 2 — Controlled fine-tuning**

| Setting | Value |
|---------|-------|
| Unfrozen backbone layers | last 30 |
| BatchNormalization layers | kept frozen |
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Max epochs | 15 (EarlyStopping patience 4) |
| Batch size | 32 |

### Model selection

The final model is selected by comparing **validation loss** from Stage 1 and
Stage 2. The lower-loss checkpoint is copied to `models/efficientnet_b0.keras`.
No test-set metric is used for selection.

### Command

```bash
# From the repository root, inside the virtual environment (WSL2)
python -m src.models.train_efficientnet
```

### Outputs

| Artefact | Location |
|----------|----------|
| Stage 1 checkpoint | `models/efficientnet_b0_frozen.keras` (**not committed**) |
| Stage 2 checkpoint | `models/efficientnet_b0_finetuned.keras` (**not committed**) |
| Final model | `models/efficientnet_b0.keras` (**not committed**) |
| Model summary | `results/efficientnet_b0/model_summary.txt` |
| Stage 1 history CSV | `results/efficientnet_b0/frozen_history.csv` |
| Stage 2 history CSV | `results/efficientnet_b0/finetune_history.csv` |
| Training curves | `results/efficientnet_b0/training_curves.png` |
| Training summary | `results/efficientnet_b0/training_summary.json` |
| Run config | `results/efficientnet_b0/run_config.json` |
| Stage comparison | `results/efficientnet_b0/stage_comparison.json` |

### Test-set evaluation

**The official Testing split is NOT evaluated in Phase 5.**
`training_summary.json` explicitly records `"test_set_evaluated": false`.
Final head-to-head comparison (baseline CNN vs. EfficientNetB0) happens in
Phase 6 using the reserved test set.

---

## Phase 6 — Final Evaluation, Model Comparison, and Grad-CAM

> **Status: ✅ Complete**

Phase 6 evaluated both pre-trained models — the Baseline Custom CNN (Phase 4) and
the EfficientNetB0 Transfer model (Phase 5) — on the official, untouched
test split (`data/processed/test_manifest.csv`, 1 600 images, 400 per class)
and generated reproducible Grad-CAM visualisations.

### Official Primary Test-Set Results

Evaluated exactly once on the full 1 600-image test set:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Selected for Phase 7 |
|-------|----------|-----------------|--------------|----------|-------------|----------------------|
| **Baseline Custom CNN** | **0.8513** (85.125%) | **0.8589** | **0.8513** | **0.8495** | **0.8495** | **Yes (Selected)** |
| **EfficientNetB0 Transfer** | 0.8431 (84.3125%) | 0.8471 | 0.8431 | 0.8366 | 0.8366 | No |

**Selection Rule:** Highest Macro F1 score on the untouched official test set.
The **Baseline CNN** achieved higher Macro F1 (0.8495 vs. 0.8366) and higher Accuracy (85.12% vs. 84.31%), and is designated as the primary model for the Phase 7 Streamlit web application.

> **Important:** Both models were evaluated exactly once on the same untouched test set. No hyperparameter tuning, retraining, or modifications were performed after opening the test set.

### Secondary Duplicate Sensitivity Analysis

Phase 2 identified **15 exact duplicate groups** inside the official Testing split, corresponding to **16 redundant copies** across 30 affected images (deduplicated sensitivity population = 1 584 images).

Sensitivity metrics were computed by filtering the in-memory prediction DataFrame without additional model inference:

| Model | Primary Test Acc (N=1600) | Primary Macro F1 (N=1600) | Dedup Test Acc (N=1584) | Dedup Macro F1 (N=1584) |
|-------|--------------------------|---------------------------|-------------------------|-------------------------|
| **Baseline Custom CNN** | 0.85125 | 0.8494955787 | **0.8573232323** | **0.8553429132** |
| **EfficientNetB0 Transfer** | 0.843125 | 0.8366468062 | 0.8491161616 | 0.8424994628 |

- Primary reported metrics use the full official 1 600-image test set.
- Sensitivity metrics are strictly secondary and demonstrate that exact duplicates within the testing split did not artificially inflate model performance.
- No additional inference or dataset reloading was performed for the sensitivity analysis.

### Grad-CAM Explainability

Grad-CAM was generated for the selected **Baseline CNN** targeting layer `block4_conv`.
To prevent cherry-picking, examples were selected using a deterministic lexicographic sort on filepaths:

- **Correctly Classified (1 per class):**
  - `correct_glioma.png`: `Te-gl_10.jpg` (confidence: 100.00%)
  - `correct_meningioma.png`: `Te-aug-me_1.jpg` (confidence: 76.34%)
  - `correct_notumor.png`: `Te-no_1.jpg` (confidence: 98.76%)
  - `correct_pituitary.png`: `Te-pi_1.jpg` (confidence: 98.01%)
- **Misclassified (First 2 lexicographical):**
  - `misclassified_01.png`: `Te-gl_1.jpg` (true: glioma, pred: notumor, confidence: 45.80%)
  - `misclassified_02.png`: `Te-gl_101.jpg` (true: glioma, pred: notumor, confidence: 72.57%)

Visualizations are structured as 3-panel figures: `Original MRI | Grad-CAM Heatmap | Overlay`.

> **Disclaimer:** Grad-CAM highlights spatial regions influencing the model's classification decision. It is **NOT** tumour segmentation and does **NOT** prove lesion location. This software is educational/research software only.

### Evaluation Artefacts (`results/final_evaluation/`)

```
results/final_evaluation/
├── baseline_metrics.json
├── efficientnet_metrics.json
├── baseline_classification_report.csv
├── efficientnet_classification_report.csv
├── baseline_predictions.csv
├── efficientnet_predictions.csv
├── baseline_confusion_matrix.csv / .png / _normalized.png
├── efficientnet_confusion_matrix.csv / .png / _normalized.png
├── model_comparison.csv
├── model_comparison.json
├── evaluation_summary.json
├── duplicate_sensitivity/
│   ├── baseline_metrics.json
│   ├── efficientnet_metrics.json
│   └── comparison.csv
└── gradcam/
    ├── correct_glioma.png
    ├── correct_meningioma.png
    ├── correct_notumor.png
    ├── correct_pituitary.png
    ├── misclassified_01.png
    ├── misclassified_02.png
    ├── gradcam_examples.csv
    └── gradcam_summary.json
```

### Educational / research disclaimer

This project performs **brain MRI image classification** for educational
and research purposes only. It is **not** a clinical diagnostic tool,
has not been clinically validated, and must not be used for medical
decision-making.

---

## License

To be determined.
