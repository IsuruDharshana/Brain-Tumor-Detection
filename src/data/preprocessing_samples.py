"""
preprocessing_samples.py
------------------------
Optional Phase 3 visual sanity check for the Brain Tumor MRI Classification
project.

Generates ``results/preprocessing/preprocessing_samples.png``:
one column per class (the lexicographically first training-manifest image),
three rows per column:

    1. original decode       (PIL, original resolution, RGB)
    2. processed 224x224     (base preprocessing: what the model would see)
    3. augmented once        (one random training augmentation, training split only)

No patient metadata is displayed (the dataset contains none). This only
visualises preprocessing - no model, no training.

How to run (from the repository root, inside the project virtual env):

    python -m src.data.preprocessing_samples
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Keep TensorFlow's C++ log noise low (set before TF is imported).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")  # headless plotting (no GUI backend required)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import CLASS_NAMES, PREPROCESSING_RESULTS_DIR, REPO_ROOT, TRAIN_MANIFEST
from src.data.data_pipeline import create_training_augmentation, read_manifest_rows
from src.data.preprocessing import load_and_preprocess_image

OUTPUT_PATH = PREPROCESSING_RESULTS_DIR / "preprocessing_samples.png"


def _first_row_per_class(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the lexicographically first training row of every class."""
    picked: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: r["relative_path"]):
        picked.setdefault(row["label_name"], row)
    missing = [label for label in CLASS_NAMES if label not in picked]
    if missing:
        raise ValueError(f"no training rows found for class(es): {missing}")
    return [picked[label] for label in CLASS_NAMES]


def main() -> int:
    """Create the preprocessing sample figure; returns a process exit code."""
    if not Path(TRAIN_MANIFEST).is_file():
        print(f"ERROR: missing train manifest: {TRAIN_MANIFEST}")
        print("       run 'python -m src.data.create_splits' first")
        return 1

    rows = _first_row_per_class(read_manifest_rows(TRAIN_MANIFEST))
    augmenter = create_training_augmentation()

    figure, axes = plt.subplots(3, len(rows), figsize=(3.2 * len(rows), 9.6))
    for column, row in enumerate(rows):
        path = Path(REPO_ROOT) / row["relative_path"]
        original = np.asarray(Image.open(path).convert("RGB"))
        processed = load_and_preprocess_image(str(path)).numpy()
        augmented = np.clip(augmenter(processed, training=True).numpy(), 0.0, 1.0)

        panels = [
            (original, f"{row['label_name']}\noriginal {original.shape[1]}x{original.shape[0]}"),
            (processed, "processed 224x224\n(float32, [0, 1])"),
            (augmented, "augmented once\n(training split only)"),
        ]
        for row_index, (image, title) in enumerate(panels):
            axis = axes[row_index][column]
            axis.imshow(image)
            axis.set_title(title, fontsize=9)
            axis.axis("off")

    figure.suptitle(
        "Phase 3 preprocessing sanity check - first training image per class",
        fontsize=11,
    )
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
