"""
test_create_splits.py
---------------------
Unit tests for Phase 3 split creation (src/data/create_splits.py) using a
tiny synthetic dataset: exact deduplication, deterministic stratified split,
manifest contents, test-set isolation and leakage detection.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.config import CLASS_NAMES, LABEL_TO_ID, RANDOM_SEED, VALIDATION_FRACTION
from src.data import create_splits
from src.data.create_splits import PreprocessingError, check_leakage
from tests.conftest import build_synthetic_repo


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha_set(rows: list[dict]) -> set:
    return {row["sha256"] for row in rows}


# ---------------------------------------------------------------------------
# Exact deduplication of the Training pool
# ---------------------------------------------------------------------------

def test_deduplication_keeps_lexicographically_first_path(synthetic_splits) -> None:
    rows = _read_rows(synthetic_splits["results_dir"] / "training_deduplication.csv")

    assert len(rows) == 2  # one excluded file in glioma, one in notumor
    by_label = {row["label_name"]: row for row in rows}

    glioma = by_label["glioma"]
    assert glioma["kept_relative_path"].endswith("Training-glioma-0.jpg")
    assert glioma["removed_relative_path"].endswith("Training-glioma-copy.jpg")

    notumor = by_label["notumor"]
    assert notumor["kept_relative_path"].endswith("Training-notumor-1.jpg")
    assert notumor["removed_relative_path"].endswith("Training-notumor-copy.jpg")


def test_summary_counts_are_derived_from_actual_data(synthetic_splits) -> None:
    summary = synthetic_splits["summary"]

    assert summary["original_training_count"] == 26          # 24 + 2 exact copies
    assert summary["removed_duplicate_count"] == 2
    assert summary["deduplicated_training_count"] == 24
    assert summary["removed_duplicates_per_class"] == {
        "glioma": 1, "meningioma": 0, "notumor": 1, "pituitary": 0,
    }
    assert summary["train_count"] == 20
    assert summary["validation_count"] == 4
    assert summary["test_count"] == 21                       # Testing kept unchanged
    assert summary["random_seed"] == RANDOM_SEED
    assert summary["validation_fraction"] == VALIDATION_FRACTION
    assert summary["label_mapping"] == LABEL_TO_ID


# ---------------------------------------------------------------------------
# Deterministic, stratified train/validation split
# ---------------------------------------------------------------------------

def test_split_is_reproducible_across_runs(synthetic_repo: Path) -> None:
    """Running the pipeline twice must reproduce identical manifests."""
    first = {
        name: (synthetic_repo / "out1" / f"{name}_manifest.csv")
        for name in ("train", "val", "test")
    }
    second = {
        name: (synthetic_repo / "out2" / f"{name}_manifest.csv")
        for name in ("train", "val", "test")
    }
    for target in (first, second):
        create_splits.run(
            data_dir=synthetic_repo / "data" / "raw",
            processed_dir=target["train"].parent,
            results_dir=synthetic_repo / "results" / "preprocessing",
            repo_root=synthetic_repo,
        )
    for name in ("train", "val", "test"):
        assert first[name].read_bytes() == second[name].read_bytes()


def test_train_val_disjoint_and_stratified(synthetic_splits) -> None:
    train_rows = _read_rows(synthetic_splits["train_manifest"])
    val_rows = _read_rows(synthetic_splits["val_manifest"])

    # 24 cleaned Training images -> 20 train / 4 validation, 1 per class after dedup
    assert len(train_rows) == 20
    assert len(val_rows) == 4

    for label in CLASS_NAMES:
        train_count = sum(1 for r in train_rows if r["label_name"] == label)
        val_count = sum(1 for r in val_rows if r["label_name"] == label)
        assert train_count == 5, label
        assert val_count == 1, label

    # no exact duplicate may be split across train and validation
    assert _sha_set(train_rows).isdisjoint(_sha_set(val_rows))
    # ... and no file either (raw files are unique after dedup inside Training)
    assert {r["relative_path"] for r in train_rows}.isdisjoint(
        {r["relative_path"] for r in val_rows}
    )


def test_label_ids_follow_fixed_mapping(synthetic_splits) -> None:
    for key in ("train_manifest", "val_manifest", "test_manifest"):
        for row in _read_rows(synthetic_splits[key]):
            assert int(row["label_id"]) == LABEL_TO_ID[row["label_name"]]


# ---------------------------------------------------------------------------
# Official Testing split stays isolated
# ---------------------------------------------------------------------------

def test_test_manifest_is_isolated_and_unchanged(synthetic_splits) -> None:
    train_rows = _read_rows(synthetic_splits["train_manifest"])
    val_rows = _read_rows(synthetic_splits["val_manifest"])
    test_rows = _read_rows(synthetic_splits["test_manifest"])

    # the official test split keeps ALL 21 files, including its duplicate pair
    assert len(test_rows) == 21
    assert all(row["source_split"] == "Testing" for row in test_rows)
    assert all(row["processed_split"] == "test" for row in test_rows)
    assert all(row["source_split"] == "Training" for row in train_rows + val_rows)

    duplicate_sha = [r["sha256"] for r in test_rows
                     if r["relative_path"].endswith("Testing-glioma-2.jpg")]
    assert len(duplicate_sha) == 1  # sanity: exactly one such path exists
    assert sum(1 for r in test_rows if r["sha256"] == duplicate_sha[0]) == 2

    # no exact hash overlap between processed train/validation and Testing
    assert _sha_set(test_rows).isdisjoint(_sha_set(train_rows) | _sha_set(val_rows))


def test_test_duplicate_metadata_for_later_sensitivity_analysis(synthetic_splits) -> None:
    rows = _read_rows(synthetic_splits["results_dir"] / "test_duplicates.csv")

    assert len(rows) == 2              # one group of two identical files
    assert {row["group_id"] for row in rows} == {"1"}
    assert all(row["group_size"] == "2" for row in rows)

    representatives = {row["relative_path"] for row in rows
                       if row["is_representative"] == "True"}
    assert len(representatives) == 1  # exactly one keeper per group

    # keeper = lexicographically first path of the group
    kept = [row["relative_path"] for row in rows if row["is_representative"] == "True"]
    removed = [row["relative_path"] for row in rows if row["is_representative"] == "False"]
    assert len(kept) == 1 and kept[0].endswith("Testing-glioma-2.jpg")
    assert len(removed) == 1 and removed[0].endswith("Testing-glioma-copy.jpg")


# ---------------------------------------------------------------------------
# Failure modes: leakage must stop the run, missing data must fail clearly
# ---------------------------------------------------------------------------

def test_cross_split_leak_aborts_without_writing_manifests(tmp_path: Path) -> None:
    build_synthetic_repo(tmp_path, cross_split_leak=True)
    processed_dir = tmp_path / "data" / "processed"

    with pytest.raises(PreprocessingError, match="leakage"):
        create_splits.run(
            data_dir=tmp_path / "data" / "raw",
            processed_dir=processed_dir,
            results_dir=tmp_path / "results" / "preprocessing",
            repo_root=tmp_path,
        )
    assert not (processed_dir / "train_manifest.csv").exists()


def test_missing_dataset_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(PreprocessingError, match="raw split folder not found"):
        create_splits.run(
            data_dir=tmp_path / "does-not-exist",
            processed_dir=tmp_path / "processed",
            results_dir=tmp_path / "results",
            repo_root=tmp_path,
        )


def test_check_leakage_unit() -> None:
    def row(sha: str) -> dict:
        return {"sha256": sha}

    assert check_leakage([row("a")], [row("b")], [row("c")]) == []

    train_val = check_leakage([row("a")], [row("a")], [row("c")])
    assert len(train_val) == 1 and "train and validation" in train_val[0]

    with_test = check_leakage([row("a")], [row("b")], [row("a")])
    assert len(with_test) == 1 and "Testing" in with_test[0]
