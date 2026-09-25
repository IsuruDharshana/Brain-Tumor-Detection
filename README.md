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
| 5 | EfficientNetB0 transfer learning | ⏳ Pending |
| 6 | Final evaluation + model comparison + Grad-CAM | ⏳ Pending |
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

## License

To be determined.
