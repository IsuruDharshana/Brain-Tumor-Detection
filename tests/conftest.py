"""
conftest.py
-----------
Shared pytest fixtures for the Phase 3 tests.

Everything here builds tiny synthetic datasets (repo-like directory trees),
so the unit tests never require the real Kaggle dataset. These fixtures are
deliberately TensorFlow-free: only the pipeline tests import TF.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.config import CLASS_NAMES


def _write_image(
    path: Path,
    color: tuple[int, ...],
    size: tuple[int, int] = (48, 32),
    mode: str = "RGB",
    image_format: str | None = None,
) -> None:
    """Save a solid-colour test image (format inferred from the extension)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path, format=image_format)


def build_synthetic_repo(repo_root: Path, cross_split_leak: bool = False) -> Path:
    """Create a repo-like tree with a tiny deterministic raw dataset.

    Layout (class colours differ by class, so no unintended duplicates):
        Training: 6 images per class + 1 exact copy in glioma (of index 0)
                  and 1 exact copy in notumor (of index 1) -> 26 files
        Testing:  5 images per class + 1 exact copy in glioma (of index 2)
                  -> 21 files (the official split is kept unchanged)

    With ``cross_split_leak=True`` an exact copy of a Training image is also
    placed inside Testing, which must make create_splits abort.

    Returns the raw data directory (``<repo_root>/data/raw``).
    """
    raw = repo_root / "data" / "raw"
    for class_index, label in enumerate(CLASS_NAMES):
        for index in range(6):
            _write_image(
                raw / "Training" / label / f"Training-{label}-{index}.jpg",
                color=(index * 35, 20 + class_index * 40, 200),
            )
        for index in range(5):
            _write_image(
                raw / "Testing" / label / f"Testing-{label}-{index}.jpg",
                color=(index * 35, 20 + class_index * 40, 150),
            )
    # exact duplicates inside Training (same bytes as the referenced images)
    _write_image(raw / "Training" / "glioma" / "Training-glioma-copy.jpg", color=(0, 20, 200))
    _write_image(raw / "Training" / "notumor" / "Training-notumor-copy.jpg", color=(35, 100, 200))
    # exact duplicate inside Testing
    _write_image(raw / "Testing" / "glioma" / "Testing-glioma-copy.jpg", color=(70, 20, 150))
    if cross_split_leak:
        # exact copy of a Training image placed inside Testing (should be caught)
        _write_image(raw / "Testing" / "glioma" / "Testing-glioma-leak.jpg", color=(0, 20, 200))
    return raw


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    """Repo-like temporary directory with a tiny synthetic raw dataset."""
    build_synthetic_repo(tmp_path)
    return tmp_path


@pytest.fixture
def synthetic_splits(synthetic_repo: Path) -> dict:
    """Run create_splits on the synthetic repo and return the output paths."""
    from src.data import create_splits

    processed_dir = synthetic_repo / "data" / "processed"
    results_dir = synthetic_repo / "results" / "preprocessing"
    summary = create_splits.run(
        data_dir=synthetic_repo / "data" / "raw",
        processed_dir=processed_dir,
        results_dir=results_dir,
        repo_root=synthetic_repo,
    )
    return {
        "repo_root": synthetic_repo,
        "processed_dir": processed_dir,
        "results_dir": results_dir,
        "train_manifest": processed_dir / "train_manifest.csv",
        "val_manifest": processed_dir / "val_manifest.csv",
        "test_manifest": processed_dir / "test_manifest.csv",
        "summary": summary,
    }
