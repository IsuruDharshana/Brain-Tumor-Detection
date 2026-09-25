"""
config.py
---------
Central Phase 3 constants for the Brain Tumor MRI Classification project.

Everything that influences reproducibility (seed, image size, split fraction,
label mapping, augmentation strengths) is defined here once, so the split
creation, the tf.data pipeline, the verification script and the tests all
work with exactly the same values.

This is intentionally a plain constants module - not a configuration
framework. Model / architecture settings (Phase 4+) do not belong here.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3  # every image is converted to RGB at load time

# float32 pixels scaled from the 0-255 integer range to [0, 1] by dividing 255.
NORMALIZATION_METHOD = "divide by 255.0 -> float32 pixel values in [0, 1]"

# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------
VALIDATION_FRACTION = 0.20  # 80% train / 20% validation, from cleaned Training

RAW_SPLIT_DIRS = ("Training", "Testing")  # original Kaggle split names

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")  # fixed order
LABEL_TO_ID = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
ID_TO_LABEL = {label_id: name for name, label_id in LABEL_TO_ID.items()}

# ---------------------------------------------------------------------------
# tf.data pipeline defaults
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 10_000  # >= training pool size -> effectively a full shuffle

# Mild, anatomically plausible augmentation (applied to the training split only)
AUG_ROTATION_DEGREES = 8.0     # ~ +/- 8 degrees (guideline: 5-10)
AUG_ZOOM_FACTOR = 0.08         # up to ~8% zoom (guideline: 5-10%)
AUG_TRANSLATION_FACTOR = 0.05  # up to ~5% translation
AUG_CONTRAST_FACTOR = 0.10     # mild contrast jitter

# ---------------------------------------------------------------------------
# Repository paths (resolved from this file, independent of the CWD)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
PREPROCESSING_RESULTS_DIR = REPO_ROOT / "results" / "preprocessing"

TRAIN_MANIFEST = PROCESSED_DATA_DIR / "train_manifest.csv"
VAL_MANIFEST = PROCESSED_DATA_DIR / "val_manifest.csv"
TEST_MANIFEST = PROCESSED_DATA_DIR / "test_manifest.csv"
