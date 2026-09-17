"""
test_data_pipeline.py
---------------------
Unit tests for the Phase 3 tf.data pipeline (src/data/data_pipeline.py):
manifest loading, batch shapes/dtypes, deterministic validation processing,
training-only augmentation and reproducible shuffling.

These tests import TensorFlow; they run on the small synthetic dataset built
by the fixtures in conftest.py, never on the full Kaggle dataset.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from src.config import CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
from src.data import data_pipeline
from src.data.data_pipeline import build_dataset, load_manifest
from src.data.preprocessing import load_and_preprocess_image

EXPECTED_IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_load_manifest_resolves_paths_and_labels(synthetic_splits) -> None:
    paths, labels = load_manifest(
        synthetic_splits["train_manifest"], root=synthetic_splits["repo_root"]
    )

    assert len(paths) == 20
    assert labels.dtype == np.int32
    assert set(labels.tolist()) == {0, 1, 2, 3}
    assert all(Path(p).is_file() for p in paths)


def test_train_batch_shapes_dtypes_and_range(synthetic_splits) -> None:
    dataset = build_dataset(
        synthetic_splits["train_manifest"],
        batch_size=8,
        training=True,
        root=synthetic_splits["repo_root"],
    )
    images, labels = next(iter(dataset))

    assert tuple(images.shape) == (8,) + EXPECTED_IMAGE_SHAPE
    assert images.dtype == tf.float32
    assert tuple(labels.shape) == (8,)
    assert labels.dtype == tf.int32
    assert float(tf.reduce_min(images)) >= 0.0
    assert float(tf.reduce_max(images)) <= 1.0


def test_validation_pipeline_is_deterministic_and_unaugmented(synthetic_splits) -> None:
    """Validation batches must equal the base preprocessing of each manifest row."""
    rows = _read_rows(synthetic_splits["val_manifest"])
    dataset = build_dataset(
        synthetic_splits["val_manifest"],
        batch_size=len(rows),
        training=False,
        root=synthetic_splits["repo_root"],
    )
    images, labels = next(iter(dataset))

    # order matches the manifest exactly (no shuffle) ...
    np.testing.assert_array_equal(labels.numpy(), [int(r["label_id"]) for r in rows])
    # ... and pixels equal the deterministic base preprocessing (no augmentation)
    first_path = Path(synthetic_splits["repo_root"]) / rows[0]["relative_path"]
    expected = load_and_preprocess_image(str(first_path)).numpy()
    np.testing.assert_allclose(images[0].numpy(), expected, atol=1e-6)


def test_augmentation_is_applied_to_training_only(synthetic_splits, monkeypatch) -> None:
    """Marker augmentation: present in the training pipeline, absent elsewhere."""

    def marker_augmentation():
        return tf.keras.Sequential(
            [tf.keras.layers.Lambda(lambda x: tf.zeros_like(x) + 0.5)]
        )

    monkeypatch.setattr(data_pipeline, "create_training_augmentation", marker_augmentation)

    train_dataset = build_dataset(
        synthetic_splits["train_manifest"],
        batch_size=8,
        training=True,
        root=synthetic_splits["repo_root"],
    )
    train_images, _ = next(iter(train_dataset))
    assert bool(tf.reduce_all(tf.abs(train_images - 0.5) < 1e-5))

    val_dataset = build_dataset(
        synthetic_splits["val_manifest"],
        batch_size=4,
        training=False,
        root=synthetic_splits["repo_root"],
    )
    val_images, _ = next(iter(val_dataset))
    assert not bool(tf.reduce_all(tf.abs(val_images - 0.5) < 1e-5))


def test_training_order_is_reproducible_with_seed(synthetic_splits) -> None:
    first = build_dataset(
        synthetic_splits["train_manifest"],
        batch_size=20,
        training=True,
        root=synthetic_splits["repo_root"],
    )
    second = build_dataset(
        synthetic_splits["train_manifest"],
        batch_size=20,
        training=True,
        root=synthetic_splits["repo_root"],
    )
    first_labels = next(iter(first))[1].numpy()
    second_labels = next(iter(second))[1].numpy()
    np.testing.assert_array_equal(first_labels, second_labels)


def test_different_shuffle_seed_changes_order(synthetic_splits) -> None:
    dataset_default = build_dataset(
        synthetic_splits["train_manifest"],
        batch_size=20,
        training=True,
        root=synthetic_splits["repo_root"],
    )
    dataset_other = build_dataset(
        synthetic_splits["train_manifest"],
        batch_size=20,
        training=True,
        shuffle_seed=7,
        root=synthetic_splits["repo_root"],
    )
    default_labels = next(iter(dataset_default))[1].numpy()
    other_labels = next(iter(dataset_other))[1].numpy()
    assert not np.array_equal(default_labels, other_labels)
