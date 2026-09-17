"""
Tests for src/data/audit_dataset.py (Phase 2 dataset audit).

The tests build tiny synthetic datasets in temporary directories, so the
full Kaggle dataset is never required to run them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import audit_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_image(path: Path, size: tuple[int, int] = (12, 9),
                color: tuple[int, int, int] = (120, 60, 200)) -> Path:
    """Create a small solid-colour RGB PNG/JPG at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _write_bytes(path: Path, payload: bytes = b"not-really-a-jpeg") -> Path:
    """Write raw (non-image) bytes to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _build_small_dataset(root: Path) -> Path:
    """Build a tiny dataset with known duplicates, a corrupt file and a text file.

    Layout (under ``root`` = data/raw equivalent):
        Training/glioma/ok1.png          Training/glioma/copy.png  (== Testing/glioma/ok1.png)
        Training/glioma/broken.jpg       (garbage bytes)
        Training/meningioma/ok2.png      Training/meningioma/copy.png (== Testing/glioma/ok2.png)
        Training/notumor/ok3.png         Training/notumor/notes.txt (non-image file)
        Training/pituitary/ok4.png
        Testing/glioma/ok1.png           Testing/glioma/ok2.png (same content, OTHER class)
        Testing/meningioma/ok5.png       Testing/notumor/ok6.png
        Testing/pituitary/ok7.png
    """
    train = root / "Training"
    test = root / "Testing"

    ok1 = _save_image(train / "glioma" / "ok1.png", color=(10, 20, 30))
    (train / "glioma" / "copy.png").write_bytes(ok1.read_bytes())          # same-split dup
    _write_bytes(train / "glioma" / "broken.jpg")                          # corrupt image file

    ok2 = _save_image(train / "meningioma" / "ok2.png", color=(200, 10, 10))
    (train / "meningioma" / "copy.png").write_bytes(ok2.read_bytes())

    _save_image(train / "notumor" / "ok3.png", color=(30, 200, 30))
    _write_bytes(train / "notumor" / "notes.txt", b"notes")                # non-image file
    _save_image(train / "pituitary" / "ok4.png", color=(40, 40, 240))

    (test / "glioma").mkdir(parents=True, exist_ok=True)
    (test / "glioma" / "ok1.png").write_bytes(ok1.read_bytes())            # cross-split dup (same class)
    (test / "glioma" / "ok2.png").write_bytes(ok2.read_bytes())            # cross-split dup (CONFLICTING class)
    _save_image(test / "meningioma" / "ok5.png", color=(240, 120, 0))
    _save_image(test / "notumor" / "ok6.png", color=(0, 120, 240))
    _save_image(test / "pituitary" / "ok7.png", color=(120, 0, 120))
    return root


def _audit_rows(root: Path) -> list[dict]:
    """Run discovery + per-file audit and return the image rows."""
    splits, _ = audit_dataset.discover_dataset(root)
    rows, _ = audit_dataset.audit_images(root, audit_dataset.collect_files(splits))
    return rows


# ---------------------------------------------------------------------------
# Hashing / metadata helpers
# ---------------------------------------------------------------------------

def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"brain-tumor-audit-test-payload" * 100
    path = _write_bytes(tmp_path / "blob.bin", payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert audit_dataset.sha256_of_file(path) == expected


def test_inspect_image_good_file(tmp_path: Path) -> None:
    path = _save_image(tmp_path / "img.png", size=(12, 9))
    info = audit_dataset.inspect_image(path)
    assert info["error"] is None
    assert (info["width"], info["height"]) == (12, 9)
    assert info["mode"] == "RGB"
    assert info["channels"] == 3
    assert info["format"] == "PNG"


def test_inspect_image_corrupt_file_is_reported_not_raised(tmp_path: Path) -> None:
    path = _write_bytes(tmp_path / "broken.jpg", b"this is not an image")
    info = audit_dataset.inspect_image(path)
    assert info["error"] is not None
    assert info["width"] is None and info["height"] is None


# ---------------------------------------------------------------------------
# Structure discovery
# ---------------------------------------------------------------------------

def test_discover_dataset_finds_expected_structure(tmp_path: Path) -> None:
    _build_small_dataset(tmp_path)
    splits, issues = audit_dataset.discover_dataset(tmp_path)
    assert set(splits) == {"Training", "Testing"}
    assert set(splits["Training"]) == {"glioma", "meningioma", "notumor", "pituitary"}
    assert issues["missing_splits"] == []
    assert issues["missing_classes"] == {}


def test_discover_dataset_reports_anomalies(tmp_path: Path) -> None:
    _save_image(tmp_path / "Training" / "glioma" / "a.png")
    _save_image(tmp_path / "Training" / "weirdclass" / "b.png")
    _save_image(tmp_path / "Testing" / "glioma" / "c.png")
    _save_image(tmp_path / "Training" / "glioma" / "nested" / "d.png")
    _write_bytes(tmp_path / "Training" / "loose_file.txt", b"x")
    _write_bytes(tmp_path / "stray.zip", b"zip")

    splits, issues = audit_dataset.discover_dataset(tmp_path)
    assert "weirdclass" in splits["Training"]
    assert "Training/weirdclass" in issues["unexpected_class_dirs"]
    assert "pituitary" in issues["missing_classes"]["Training"]
    assert "Training/loose_file.txt" in issues["unexpected_files_in_split_dirs"]
    assert issues["unexpected_root_files"] == ["stray.zip"]
    assert issues["nested_dirs_in_class_dirs"] == [
        {"path": "Training/glioma/nested", "file_count": 1}
    ]


def test_discover_dataset_fails_when_root_missing(tmp_path: Path) -> None:
    with pytest.raises(audit_dataset.DatasetAuditError):
        audit_dataset.discover_dataset(tmp_path / "does-not-exist")


def test_discover_dataset_fails_without_expected_splits(tmp_path: Path) -> None:
    (tmp_path / "random_dir").mkdir()
    with pytest.raises(audit_dataset.DatasetAuditError):
        audit_dataset.discover_dataset(tmp_path)


# ---------------------------------------------------------------------------
# Duplicate logic
# ---------------------------------------------------------------------------

def _row(path: str, split: str, class_name: str, digest: str, valid: bool = True) -> dict:
    return {
        "relative_path": path, "split": split, "class": class_name,
        "extension": ".png", "width": 10, "height": 10, "mode": "RGB",
        "channels": 3, "format": "PNG", "file_size_bytes": 100,
        "sha256": digest, "valid": valid, "error": None,
    }


def test_find_exact_duplicates_groups_and_classifies() -> None:
    rows = [
        _row("Training/glioma/a.png", "Training", "glioma", "aaa"),
        _row("Testing/glioma/a.png", "Testing", "glioma", "aaa"),          # cross-split, same class
        _row("Training/meningioma/b.png", "Training", "meningioma", "bbb"),
        _row("Training/glioma/b2.png", "Training", "glioma", "bbb"),       # same split, CONFLICTING class
        _row("Training/notumor/unique.png", "Training", "notumor", "ccc"),
        _row("Training/notumor/bad.png", "Training", "notumor", "ddd", valid=False),
    ]
    groups = audit_dataset.find_exact_duplicates(rows)
    assert set(groups) == {"aaa", "bbb"}                      # single files / invalid excluded
    assert audit_dataset.group_type(groups["aaa"]) == "same-class"
    assert audit_dataset.group_type(groups["bbb"]) == "conflicting-class"

    stats = audit_dataset.summarise_duplicates(groups, valid_images=5)
    assert stats["groups_total"] == 2
    assert stats["files_in_duplicate_groups"] == 4
    assert stats["groups_crossing_splits"] == 1
    assert stats["cross_split_files_training"] == 1
    assert stats["cross_split_files_testing"] == 1
    assert stats["conflicting_class_groups"] == 1
    assert stats["conflicting_class_files"] == 2

    cross_rows = audit_dataset.duplicate_rows(groups, only_cross_split=True)
    assert {row["sha256"] for row in cross_rows} == {"aaa"}
    assert all(row["crosses_splits"] for row in cross_rows)


def test_hamming_pairs_and_grouping() -> None:
    # 0b0000..., 0b0000...0001, 0b00..011 (distance chain)
    values = np.array([0, 1, 3, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    pairs, capped = audit_dataset.hamming_pairs(values, threshold=2)
    assert not capped
    assert pairs == [(0, 1), (0, 2), (1, 2)]                  # i < j only, far hash excluded
    groups = audit_dataset.group_indices_by_pairs(len(values), pairs)
    assert groups == [[0, 1, 2]]


# ---------------------------------------------------------------------------
# Perceptual analysis
# ---------------------------------------------------------------------------

def test_perceptual_analysis_finds_identical_content(tmp_path: Path) -> None:
    pytest.importorskip("imagehash")
    original = _save_image(tmp_path / "Training" / "glioma" / "g1.png")
    (tmp_path / "Testing" / "glioma").mkdir(parents=True, exist_ok=True)
    (tmp_path / "Testing" / "glioma" / "g1_copy.png").write_bytes(original.read_bytes())

    rows = _audit_rows(tmp_path)
    stats, out_rows = audit_dataset.run_perceptual_analysis(tmp_path, rows)

    assert stats["status"] == "completed"
    assert stats["groups"] == 1
    assert stats["files_in_groups"] == 2
    assert stats["cross_split_groups"] == 1
    assert len(out_rows) == 2


# ---------------------------------------------------------------------------
# End-to-end on a tiny synthetic dataset
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline_and_outputs(tmp_path: Path) -> None:
    root = _build_small_dataset(tmp_path / "raw")
    out_dir = tmp_path / "out"
    files_before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))

    exit_code = audit_dataset.main([
        "--data-dir", str(root), "--output-dir", str(out_dir), "--skip-perceptual",
    ])
    assert exit_code == 0

    # The audit must not modify the dataset.
    files_after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    assert files_before == files_after

    for name in ("audit_report.md", "dataset_summary.json", "class_distribution.csv",
                 "image_metadata.csv", "corrupt_images.csv", "exact_duplicates.csv",
                 "cross_split_duplicates.csv", "class_distribution.png",
                 "image_dimension_distribution.png"):
        assert (out_dir / name).exists(), f"missing output: {name}"

    summary = json.loads((out_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    no_image_rows = [f for f in files_before if f.endswith((".png", ".jpg"))]
    assert summary["counts"]["total_images"] == len(no_image_rows)
    assert summary["counts"]["invalid_images"] == 1                    # broken.jpg
    assert summary["counts"]["non_image_files"] == 1                   # notes.txt
    assert summary["counts"]["per_split"]["Training"] == 7
    assert summary["counts"]["per_split"]["Testing"] == 5

    duplicates = summary["duplicates"]
    assert duplicates["groups_total"] == 2                             # ok1 pair + ok2 pair
    assert duplicates["groups_crossing_splits"] == 2
    assert duplicates["cross_split_files_training"] == 4
    assert duplicates["cross_split_files_testing"] == 2
    assert duplicates["conflicting_class_groups"] == 1                 # ok2: glioma vs meningioma

    report = (out_dir / "audit_report.md").read_text(encoding="utf-8")
    assert "# Dataset Audit Report" in report
    assert "Cross-split duplicates" in report


def test_main_fails_cleanly_when_dataset_missing(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    exit_code = audit_dataset.main([
        "--data-dir", str(tmp_path / "nope"), "--output-dir", str(out_dir),
    ])
    assert exit_code == 1
    assert not out_dir.exists()
