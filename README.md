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

## Running Tests

```powershell
pytest tests/
```

---

## License

To be determined.
