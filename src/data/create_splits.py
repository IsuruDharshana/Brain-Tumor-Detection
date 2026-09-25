"""
create_splits.py
----------------
Phase 3 split creation for the Brain Tumor MRI Classification project.

Builds deterministic train / validation / test manifests from the RAW,
IMMUTABLE dataset in ``data/raw``. Nothing inside ``data/raw`` is ever
modified, renamed, resized or deleted - every cleaning decision is expressed
only through the generated manifests and summaries.

Policy implemented here (detailed in the generated
``results/preprocessing/split_summary.json``):

1. Exact deduplication of the TRAINING pool only, using SHA-256 content
   hashes. Inside every duplicate group exactly one representative is kept:
   the lexicographically first normalised relative path. The other members
   are excluded from the processed training pool (they stay untouched on
   disk) and are listed in ``training_deduplication.csv``.
2. Deterministic, stratified 80/20 train/validation split of the cleaned
   Training pool (seed 42, per-class shuffle via ``random.Random``).
3. The original Kaggle Testing split is kept fully isolated and unchanged,
   even though it contains 15 exact duplicate groups. Those are recorded
   separately in ``test_duplicates.csv`` so Phase 6 can later run a
   deduplicated-test sensitivity analysis without touching the official test
   split that is used for the primary evaluation.
4. Safety checks: no SHA-256 hash may appear in two processed splits, and no
   hash may overlap between processed train/validation and the original
   Testing split. Leakage aborts with an error instead of continuing
   silently.

How to run (from the repository root, inside the project virtual env):

    python -m src.data.create_splits

Running the script twice on an unchanged dataset reproduces identical
manifests.

Outputs
    data/processed/train_manifest.csv
    data/processed/val_manifest.csv
    data/processed/test_manifest.csv
    results/preprocessing/training_deduplication.csv
    results/preprocessing/test_duplicates.csv
    results/preprocessing/split_summary.json
    results/preprocessing/class_distribution.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.config import (
    CHANNELS,
    CLASS_NAMES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABEL_TO_ID,
    NORMALIZATION_METHOD,
    PREPROCESSING_RESULTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    RAW_SPLIT_DIRS,
    REPO_ROOT,
    VALIDATION_FRACTION,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HASH_CHUNK_SIZE = 1 << 20  # 1 MiB chunks while hashing (keeps RAM flat)
PROGRESS_EVERY = 500       # print progress every N hashed files
MAX_EXAMPLE_HASHES = 3     # example hashes embedded in leakage error messages

MANIFEST_COLUMNS = [
    "relative_path", "label_name", "label_id",
    "source_split", "processed_split", "sha256",
]
DEDUPLICATION_COLUMNS = ["sha256", "kept_relative_path", "removed_relative_path", "label_name"]
TEST_DUPLICATE_COLUMNS = [
    "group_id", "sha256", "relative_path", "label_name", "is_representative", "group_size",
]
CLASS_DISTRIBUTION_COLUMNS = ["processed_split", "label_name", "image_count", "percentage"]

# Same image extensions as the Phase 2 audit; anything else in a class folder
# is reported and skipped (the audit verified there should be none).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

DEDUPLICATION_POLICY = (
    "inside each exact SHA-256 duplicate group of the Training pool, keep the "
    "lexicographically first normalised relative path and exclude every other "
    "member from the processed training pool (raw files stay untouched)"
)
SPLIT_POLICY = (
    "stratified per-class split of the cleaned Training pool: each class list "
    "is sorted by relative path, shuffled with a single random.Random(seed) "
    "used in the fixed CLASS_NAMES order, and the first "
    "round(n * validation_fraction) entries become validation; the original "
    "Testing split is kept fully isolated"
)


class PreprocessingError(Exception):
    """The dataset is missing or the produced split is unusable (Phase 3 must stop)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Return the SHA-256 hex digest of a file, read in fixed-size chunks.

    Same chunked approach as the Phase 2 audit, so both phases agree on every
    content hash.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalised_relative_path(path: Path, root: Path) -> str:
    """Portable repo-root-relative path with forward slashes ('data/raw/...').

    Manifests use this form everywhere instead of machine-specific absolute
    paths, so they stay valid on the WSL2 training workstation.
    """
    return path.relative_to(root).as_posix()


def per_class_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count rows per class in the fixed CLASS_NAMES order."""
    rows = list(rows)
    return {label: sum(1 for row in rows if row["label_name"] == label) for label in CLASS_NAMES}


# ---------------------------------------------------------------------------
# Step 1: collect + hash every raw image
# ---------------------------------------------------------------------------

def collect_image_rows(data_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    """Inspect the expected raw folder structure and hash every image.

    Returns one dict per image: relative_path, label_name, label_id,
    source_split, sha256 (plus the internal file_path, dropped when writing).
    Fails clearly when the raw dataset is missing.
    """
    rows: list[dict[str, Any]] = []
    for split in RAW_SPLIT_DIRS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise PreprocessingError(
                f"expected raw split folder not found: {split_dir} - "
                "download the dataset first (see README 'Dataset')"
            )
        for label in CLASS_NAMES:
            class_dir = split_dir / label
            if not class_dir.is_dir():
                raise PreprocessingError(f"expected class folder not found: {class_dir}")
            files = sorted(p for p in class_dir.iterdir() if p.is_file())
            for path in files:
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    print(f"  [warn] skipping non-image file: "
                          f"{normalised_relative_path(path, repo_root)}")
                    continue
                rows.append({
                    "relative_path": normalised_relative_path(path, repo_root),
                    "label_name": label,
                    "label_id": LABEL_TO_ID[label],
                    "source_split": split,
                    "sha256": sha256_of_file(path),
                    "file_path": path,
                })
                if len(rows) % PROGRESS_EVERY == 0:
                    print(f"  hashed {len(rows)} images ...")
    return rows


# ---------------------------------------------------------------------------
# Step 2: exact deduplication of the Training pool
# ---------------------------------------------------------------------------

def deduplicate_training(
    training_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split Training rows into kept representatives and excluded duplicates.

    Images that are unique are their own group of one and always kept.
    Equality is on exact SHA-256 content hash. The keeper of a duplicate
    group is the lexicographically first normalised relative path (plain
    string order, not a natural/numeric sort).
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in training_rows:
        groups[row["sha256"]].append(row)

    kept_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    for sha, members in groups.items():
        members_sorted = sorted(members, key=lambda r: r["relative_path"])
        keeper = members_sorted[0]
        kept_rows.append(keeper)
        for duplicate in members_sorted[1:]:
            removed_rows.append({
                "sha256": sha,
                "kept_relative_path": keeper["relative_path"],
                "removed_relative_path": duplicate["relative_path"],
                "label_name": duplicate["label_name"],
            })
    return kept_rows, removed_rows


# ---------------------------------------------------------------------------
# Step 3: deterministic stratified train/validation split
# ---------------------------------------------------------------------------

def stratified_train_val_split(
    kept_rows: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the cleaned Training pool into train / validation, stratified by class.

    Each class is handled independently: sort by relative path, shuffle with
    one shared seeded RNG (classes processed strictly in CLASS_NAMES order),
    then the first ``round(n * val_fraction)`` entries become validation.
    Because the input is sorted and the RNG is seeded, the same dataset and
    code always reproduce the same split.
    """
    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for label in CLASS_NAMES:
        class_rows = sorted(
            (row for row in kept_rows if row["label_name"] == label),
            key=lambda r: r["relative_path"],
        )
        rng.shuffle(class_rows)
        n_val = int(round(len(class_rows) * val_fraction))
        val_rows.extend(class_rows[:n_val])
        train_rows.extend(class_rows[n_val:])
    return train_rows, val_rows


# ---------------------------------------------------------------------------
# Step 4: leakage checks (stop rather than continue silently)
# ---------------------------------------------------------------------------

def check_leakage(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[str]:
    """Return human-readable leakage problems; an empty list means clean.

    Checked with final manifests/hash sets (not expectations):
      * no SHA-256 hash in both train and validation
      * no SHA-256 hash overlap between processed train/validation and Testing
    """
    problems: list[str] = []

    train_hashes = {row["sha256"] for row in train_rows}
    val_hashes = {row["sha256"] for row in val_rows}
    test_hashes = {row["sha256"] for row in test_rows}

    train_val = train_hashes & val_hashes
    if train_val:
        examples = ", ".join(sorted(train_val)[:MAX_EXAMPLE_HASHES])
        problems.append(
            f"{len(train_val)} SHA-256 hash(es) appear in both train and validation "
            f"(e.g. {examples})"
        )

    train_val_test = (train_hashes | val_hashes) & test_hashes
    if train_val_test:
        examples = ", ".join(sorted(train_val_test)[:MAX_EXAMPLE_HASHES])
        problems.append(
            f"{len(train_val_test)} SHA-256 hash(es) overlap between processed "
            f"train/validation and the original Testing split (e.g. {examples})"
        )

    return problems


# ---------------------------------------------------------------------------
# Step 5: output writers
# ---------------------------------------------------------------------------

def write_manifest(path: Path, rows: list[dict[str, Any]], processed_split: str) -> None:
    """Write one manifest CSV, sorted by relative path for stable diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["relative_path"]):
            writer.writerow({
                "relative_path": row["relative_path"],
                "label_name": row["label_name"],
                "label_id": row["label_id"],
                "source_split": row["source_split"],
                "processed_split": processed_split,
                "sha256": row["sha256"],
            })


def write_deduplication_report(path: Path, removed_rows: list[dict[str, Any]]) -> None:
    """Write one row per excluded Training duplicate (kept vs removed path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEDUPLICATION_COLUMNS)
        writer.writeheader()
        for row in sorted(
            removed_rows,
            key=lambda r: (r["kept_relative_path"], r["removed_relative_path"]),
        ):
            writer.writerow(row)


def write_test_duplicate_metadata(path: Path, test_rows: list[dict[str, Any]]) -> None:
    """Record exact duplicate groups inside Testing (official test split unchanged).

    Every member row is listed with a group_id; ``is_representative`` marks
    the lexicographically first path of the group. Phase 6 can build a
    deduplicated test variant by keeping representative rows only - without
    ever modifying the primary test manifest.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        groups[row["sha256"]].append(row)
    duplicate_groups = [members for members in groups.values() if len(members) > 1]
    ordered_groups = sorted(
        duplicate_groups,
        key=lambda members: sorted(r["relative_path"] for r in members)[0],
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEST_DUPLICATE_COLUMNS)
        writer.writeheader()
        for group_id, members in enumerate(ordered_groups, start=1):
            members_sorted = sorted(members, key=lambda r: r["relative_path"])
            for position, row in enumerate(members_sorted):
                writer.writerow({
                    "group_id": group_id,
                    "sha256": row["sha256"],
                    "relative_path": row["relative_path"],
                    "label_name": row["label_name"],
                    "is_representative": position == 0,
                    "group_size": len(members_sorted),
                })


def write_class_distribution(
    path: Path,
    split_rows: dict[str, list[dict[str, Any]]],
) -> None:
    """Write per-class counts and percentages for train / validation / test."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLASS_DISTRIBUTION_COLUMNS)
        writer.writeheader()
        for processed_split in ("train", "validation", "test"):
            rows = split_rows[processed_split]
            total = len(rows)
            counts = per_class_counts(rows)
            for label in CLASS_NAMES:
                percentage = round(100.0 * counts[label] / total, 2) if total else 0.0
                writer.writerow({
                    "processed_split": processed_split,
                    "label_name": label,
                    "image_count": counts[label],
                    "percentage": percentage,
                })


def build_summary(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    removed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the split_summary.json content from actual data."""
    removed_per_class = per_class_counts(removed_rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": "Phase 3 - preprocessing and data input pipeline",
        "random_seed": RANDOM_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "image_size": {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
        "channels": CHANNELS,
        "normalization_method": NORMALIZATION_METHOD,
        "label_mapping": dict(LABEL_TO_ID),
        "original_training_count": len(train_rows) + len(val_rows) + len(removed_rows),
        "deduplicated_training_count": len(train_rows) + len(val_rows),
        "removed_duplicate_count": len(removed_rows),
        "removed_duplicates_per_class": removed_per_class,
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "test_count": len(test_rows),
        "class_counts": {
            "train": per_class_counts(train_rows),
            "validation": per_class_counts(val_rows),
            "test": per_class_counts(test_rows),
        },
        "deduplication_policy": DEDUPLICATION_POLICY,
        "split_policy": SPLIT_POLICY,
        "test_duplicates_note": (
            "the original Testing split is kept unchanged in the primary test "
            "manifest; its internal exact duplicates are listed in "
            "test_duplicates.csv for a Phase 6 sensitivity analysis"
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Step 6: optional cross-check against the Phase 2 audit artefacts
# ---------------------------------------------------------------------------

def cross_check_phase2_audit(
    repo_root: Path,
    duplicate_group_count: int,
    duplicate_member_count: int,
) -> None:
    """Compare our recomputed Training duplicates with the Phase 2 audit CSV.

    ``duplicate_member_count`` counts every Training file that belongs to a
    duplicate group, including the kept representative - this is exactly what
    the Phase 2 audit lists in exact_duplicates.csv (one row per member file).

    This is a consistency note only (fail-soft): the numbers computed in this
    run stay authoritative, but a mismatch is worth a human look.
    """
    audit_csv = repo_root / "results" / "dataset_audit" / "exact_duplicates.csv"
    if not audit_csv.is_file():
        print("  Phase 2 audit CSV not found - skipping duplicates cross-check")
        return

    with audit_csv.open(newline="", encoding="utf-8") as handle:
        audit_rows = [
            row for row in csv.DictReader(handle) if row.get("split") == "Training"
        ]
    audit_members = len(audit_rows)
    audit_groups = len({row["sha256"] for row in audit_rows})

    if (audit_groups, audit_members) == (duplicate_group_count, duplicate_member_count):
        print(
            f"  cross-check OK: recomputed Training duplicates match the Phase 2 "
            f"audit ({duplicate_group_count} groups / {duplicate_member_count} files)"
        )
    else:
        print(
            f"  [warn] Training duplicates differ from the Phase 2 audit: "
            f"now {duplicate_group_count} groups / {duplicate_member_count} files, "
            f"audit had {audit_groups} groups / {audit_members} files - please review"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    data_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    results_dir: Path = PREPROCESSING_RESULTS_DIR,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Create manifests + summaries. Raises PreprocessingError on any blocker."""
    data_dir = Path(data_dir).resolve()
    processed_dir = Path(processed_dir).resolve()
    results_dir = Path(results_dir).resolve()
    repo_root = Path(repo_root).resolve()

    print("Phase 3 - deterministic train/validation/test split creation")
    print("  (data/raw is read-only; cleaning is expressed only through manifests)")
    print(f"  raw data dir : {data_dir}")

    # 1. hash everything (raw stays untouched)
    rows = collect_image_rows(data_dir, repo_root)
    training_rows = [r for r in rows if r["source_split"] == "Training"]
    test_rows = [r for r in rows if r["source_split"] == "Testing"]
    if not training_rows or not test_rows:
        raise PreprocessingError(
            f"expected images in both raw splits, found "
            f"Training={len(training_rows)}, Testing={len(test_rows)}"
        )
    print(f"  hashed {len(rows)} images "
          f"({len(training_rows)} Training / {len(test_rows)} Testing)")

    # 2. exact deduplication of the Training pool
    kept_rows, removed_rows = deduplicate_training(training_rows)
    duplicate_group_count = len({r["sha256"] for r in removed_rows})
    duplicate_member_count = duplicate_group_count + len(removed_rows)
    print(f"  exact duplicates excluded from Training: {len(removed_rows)} files "
          f"in {duplicate_group_count} groups -> cleaned pool {len(kept_rows)}")

    # 3. stratified 80/20 split of the cleaned pool
    train_rows, val_rows = stratified_train_val_split(
        kept_rows, VALIDATION_FRACTION, RANDOM_SEED
    )
    print(f"  split: {len(train_rows)} train / {len(val_rows)} validation "
          f"(seed {RANDOM_SEED}, validation fraction {VALIDATION_FRACTION})")

    # 4. leakage checks - stop instead of continuing silently
    problems = check_leakage(train_rows, val_rows, test_rows)
    if problems:
        for problem in problems:
            print(f"  [FAIL] {problem}")
        raise PreprocessingError(
            "exact-duplicate leakage detected between processed splits - "
            "stopping before writing manifests"
        )
    print("  leakage checks passed: no train/validation hash overlap, "
          "no overlap with the original Testing split")

    # 5. write all outputs
    write_manifest(processed_dir / "train_manifest.csv", train_rows, "train")
    write_manifest(processed_dir / "val_manifest.csv", val_rows, "validation")
    write_manifest(processed_dir / "test_manifest.csv", test_rows, "test")
    write_deduplication_report(results_dir / "training_deduplication.csv", removed_rows)
    write_test_duplicate_metadata(results_dir / "test_duplicates.csv", test_rows)

    summary = build_summary(train_rows, val_rows, test_rows, removed_rows)
    write_json(results_dir / "split_summary.json", summary)
    write_class_distribution(
        results_dir / "class_distribution.csv",
        {"train": train_rows, "validation": val_rows, "test": test_rows},
    )

    # 6. consistency note against Phase 2
    cross_check_phase2_audit(repo_root, duplicate_group_count, duplicate_member_count)

    print("  outputs written:")
    print(f"    {processed_dir / 'train_manifest.csv'} ({len(train_rows)} rows)")
    print(f"    {processed_dir / 'val_manifest.csv'} ({len(val_rows)} rows)")
    print(f"    {processed_dir / 'test_manifest.csv'} ({len(test_rows)} rows)")
    print(f"    {results_dir / 'training_deduplication.csv'} ({len(removed_rows)} rows)")
    print(f"    {results_dir / 'test_duplicates.csv'} (Testing duplicates, official split untouched)")
    print(f"    {results_dir / 'split_summary.json'}")
    print(f"    {results_dir / 'class_distribution.csv'}")
    print("  per-class counts:")
    for name, split in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
        counts = per_class_counts(split)
        details = ", ".join(f"{label} {counts[label]}" for label in CLASS_NAMES)
        print(f"    {name:<10} {len(split)}  ({details})")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: python -m src.data.create_splits"""
    parser = argparse.ArgumentParser(
        description="Phase 3: create deterministic train/validation/test manifests "
                    "from the raw dataset (read-only) plus deduplication and split summaries."
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR,
                        help="raw dataset directory (default: data/raw)")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR,
                        help="manifest output directory (default: data/processed)")
    parser.add_argument("--results-dir", type=Path, default=PREPROCESSING_RESULTS_DIR,
                        help="summary output directory (default: results/preprocessing)")
    args = parser.parse_args(argv)

    try:
        run(data_dir=args.data_dir, processed_dir=args.processed_dir, results_dir=args.results_dir)
    except PreprocessingError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
