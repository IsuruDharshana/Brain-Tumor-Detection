"""
verify_pipeline.py
------------------
Phase 3 lightweight pipeline verification for the Brain Tumor MRI
Classification project. NO training happens here.

Steps
    1. load the train/validation/test manifests
    2. re-run the exact-duplicate leakage checks on the final manifests
       (train vs validation, and train/validation vs the original Testing)
    3. build each tf.data pipeline and retrieve ONE batch
    4. print batch shapes, dtypes, pixel range, label ids and GPU availability

How to run (from the repository root, inside the project virtual env):

    python -m src.data.verify_pipeline
    python -m src.data.verify_pipeline --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep TensorFlow's C++ log noise low (set before TF is imported).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from src.config import (
    DEFAULT_BATCH_SIZE,
    ID_TO_LABEL,
    TEST_MANIFEST,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)
from src.data.create_splits import check_leakage
from src.data.data_pipeline import build_dataset, read_manifest_rows


def main(argv: list[str] | None = None) -> int:
    """Verify manifests and pipelines; returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Phase 3: verify manifests and tf.data pipelines (no training)."
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=VAL_MANIFEST)
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST)
    args = parser.parse_args(argv)

    manifests = {
        "train": args.train_manifest,
        "validation": args.val_manifest,
        "test": args.test_manifest,
    }
    for name, path in manifests.items():
        if not Path(path).is_file():
            print(f"ERROR: missing {name} manifest: {path}")
            print("       run 'python -m src.data.create_splits' first")
            return 1

    print("Phase 3 pipeline verification (no training)")

    # 1. leakage checks on the final manifests
    train_rows = read_manifest_rows(args.train_manifest)
    val_rows = read_manifest_rows(args.val_manifest)
    test_rows = read_manifest_rows(args.test_manifest)
    problems = check_leakage(train_rows, val_rows, test_rows)
    if problems:
        for problem in problems:
            print(f"  [FAIL] {problem}")
        print("ERROR: exact-duplicate leakage detected in the final manifests - stopping")
        return 1
    print("  leakage checks passed (train/validation disjoint; no overlap with Testing)")
    print(f"  manifest sizes: train {len(train_rows)}, validation {len(val_rows)}, "
          f"test {len(test_rows)}")

    # 2. environment info (informational only)
    print(f"  TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPU devices: {[gpu.name for gpu in gpus] if gpus else 'none (CPU-only verification)'}")

    # 3. one batch per split
    for name, path in manifests.items():
        dataset = build_dataset(path, batch_size=args.batch_size, training=(name == "train"))
        images, labels = next(iter(dataset))
        label_ids = sorted({int(value) for value in labels.numpy()})
        label_names = ", ".join(ID_TO_LABEL[value] for value in label_ids)
        print(f"\n{name}:")
        print(f"  images shape: {tuple(images.shape)}   dtype: {images.dtype.name}")
        print(f"  labels shape: {tuple(labels.shape)}   dtype: {labels.dtype.name}")
        print(f"  pixel range : {float(tf.reduce_min(images)):.4f} - "
              f"{float(tf.reduce_max(images)):.4f}")
        print(f"  label ids in batch: {label_ids}  ({label_names})")

    print("\nExpected: images (batch, 224, 224, 3) float32, labels (batch,) int32, "
          "pixels inside [0, 1]. Training batches are shuffled + augmented; "
          "validation/test batches are deterministic and unaugmented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
