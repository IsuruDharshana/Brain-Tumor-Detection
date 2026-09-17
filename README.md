# Brain Tumor MRI Classification

A university machine-learning project that trains a Convolutional Neural
Network (CNN) to classify brain MRI scans into tumour categories.

---

## Current Purpose (Phase 1)

Set up a clean, portable Python project environment on the development
laptop (Intel Core i5 12th Gen · 8 GB RAM · no dedicated GPU).

Full deep-learning **model training** will be performed on a separate
**NVIDIA RTX 3060 12 GB / 32 GB RAM** workstation in a later phase.

---

## Project Structure

```
brain-tumor-classification/
├── app/                  # Streamlit web UI (Phase 3+)
├── data/
│   ├── raw/              # Original, unmodified dataset (not committed)
│   └── processed/        # Pre-processed images / splits (not committed)
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

## Running Tests

```powershell
python -m pytest
```

---

## License

To be determined.
