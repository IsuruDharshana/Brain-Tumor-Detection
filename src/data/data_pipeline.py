"""
data_pipeline.py
----------------
Phase 3 tf.data input pipeline for the Brain Tumor MRI Classification project.

Builds batched ``tf.data.Dataset`` objects of (images, integer labels) from
the manifests created by ``src.data.create_splits``:

    Training:            manifest -> (path, label) -> shuffle -> decode -> RGB
                         -> resize -> normalize -> augmentation -> batch -> prefetch

    Validation / test:   manifest -> (path, label) -> decode -> RGB -> resize
                         -> normalize -> batch -> prefetch
                         (no shuffle, no augmentation, deterministic order)

Note on shuffle placement: for per-sample transformations, shuffling the cheap
(path, label) tuples before decoding is statistically equivalent to shuffling
the decoded+augmented tensors afterwards, but it avoids buffering thousands of
224x224x3 float32 images (hundreds of MB) in the shuffle buffer. The seed
makes the batch order reproducible, and it is resampled for every epoch.

Labels stay integer (glioma=0, meningioma=1, notumor=2, pituitary=3) so the
future model can use ``SparseCategoricalCrossentropy`` directly.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

# Keep TensorFlow's C++ log noise low (set before TF is imported).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from src.config import (
    AUG_CONTRAST_FACTOR,
    AUG_ROTATION_DEGREES,
    AUG_TRANSLATION_FACTOR,
    AUG_ZOOM_FACTOR,
    DEFAULT_BATCH_SIZE,
    RANDOM_SEED,
    REPO_ROOT,
    SHUFFLE_BUFFER_SIZE,
)
from src.data.preprocessing import load_and_preprocess_image


def read_manifest_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Return the raw rows of a manifest CSV (dicts, keeps file order).

    Used by verification/visualisation tools; the training pipeline itself
    only needs ``load_manifest``.
    """
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_manifest(
    manifest_path: str | Path,
    root: str | Path = REPO_ROOT,
) -> tuple[list[str], np.ndarray]:
    """Read a manifest CSV and return (absolute image paths, int32 labels).

    Manifests store images as repo-root-relative POSIX paths
    ("data/raw/Training/glioma/...jpg"); they are resolved against ``root``
    here so the same manifests work on the Windows laptop and the WSL2
    training workstation.
    """
    paths: list[str] = []
    labels: list[int] = []
    for row in read_manifest_rows(manifest_path):
        relative_path = (row.get("relative_path") or "").strip()
        if not relative_path:
            continue
        paths.append(str(Path(root) / Path(relative_path)))
        labels.append(int(row["label_id"]))
    if not paths:
        raise ValueError(f"manifest is empty or unreadable: {manifest_path}")
    return paths, np.asarray(labels, dtype=np.int32)


def create_training_augmentation() -> tf.keras.Sequential:
    """Mild, anatomically plausible augmentation (training split only).

    Deliberately conservative: rotation ~ +/-8 degrees, zoom <= 8%,
    translation <= 5%, mild contrast jitter. No horizontal/vertical flips,
    shear, crops or colour shifts - MRI left/right orientation can carry
    meaning. Areas exposed by rotation/zoom/translation are filled black, as
    in the original MRI background.

    Every layer is created with the integer ``RANDOM_SEED``: Keras wraps it
    in a stateful SeedGenerator, so the random sequence is reproducible
    across runs while each call still receives fresh random values (the
    augmentation varies for every image and every epoch).
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=AUG_ROTATION_DEGREES / 360.0,
                fill_mode="constant",
                fill_value=0.0,
                seed=RANDOM_SEED,
                name="aug_random_rotation",
            ),
            tf.keras.layers.RandomZoom(
                height_factor=AUG_ZOOM_FACTOR,
                width_factor=AUG_ZOOM_FACTOR,
                fill_mode="constant",
                fill_value=0.0,
                seed=RANDOM_SEED,
                name="aug_random_zoom",
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=AUG_TRANSLATION_FACTOR,
                width_factor=AUG_TRANSLATION_FACTOR,
                fill_mode="constant",
                fill_value=0.0,
                seed=RANDOM_SEED,
                name="aug_random_translation",
            ),
            tf.keras.layers.RandomContrast(
                factor=AUG_CONTRAST_FACTOR,
                seed=RANDOM_SEED,
                name="aug_random_contrast",
            ),
        ],
        name="training_augmentation",
    )


def build_dataset(
    manifest_path: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    training: bool = False,
    shuffle_seed: int = RANDOM_SEED,
    root: str | Path = REPO_ROOT,
) -> tf.data.Dataset:
    """Build a batched tf.data.Dataset of (images, labels) from one manifest.

    Args:
        manifest_path: manifest CSV produced by src.data.create_splits.
        batch_size: batch size (default 32).
        training: True adds shuffling and mild augmentation; validation/test
            must use False so their inputs stay deterministic and unaugmented.
        shuffle_seed: seed for the training shuffle (ignored when not training).
        root: directory the manifest's relative paths are resolved against
            (defaults to the repository root; tests override it).

    Returns:
        tf.data.Dataset yielding (float32 images [B, 224, 224, 3],
        int32 labels [B]).
    """
    paths, labels = load_manifest(manifest_path, root=root)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        # Shuffle the small (path, label) tuples first - see module docstring.
        buffer_size = min(len(paths), SHUFFLE_BUFFER_SIZE)
        dataset = dataset.shuffle(
            buffer_size=buffer_size, seed=shuffle_seed, reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        lambda path, label: (load_and_preprocess_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        augmenter = create_training_augmentation()
        dataset = dataset.map(
            lambda image, label: (
                # Clip back to [0, 1]: contrast jitter can slightly overshoot.
                tf.clip_by_value(augmenter(image, training=True), 0.0, 1.0),
                label,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
