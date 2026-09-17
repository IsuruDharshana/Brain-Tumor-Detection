"""
audit_dataset.py
----------------
Phase 2 dataset audit for the Brain Tumor MRI Classification project.

Performs a READ-ONLY audit of the raw Kaggle dataset stored in ``data/raw``
and produces reproducible summary artefacts under ``results/dataset_audit``.
It never modifies, renames, resizes or deletes anything inside the dataset.

What is checked
    * directory structure (splits, class folders, unexpected entries)
    * image counts per split and class, plus class-imbalance statistics
    * file extensions (image files vs unexpected non-image files)
    * image integrity (corrupt / unreadable files)
    * image dimensions, colour modes and channel counts
    * exact duplicates via SHA-256 content hashes (within and across splits)
    * cross-split (Training <-> Testing) duplicates -> potential leakage
    * optional perceptual (pHash) near-duplicate analysis

How to run (from the repository root, inside the project virtual env):

    python -m src.data.audit_dataset
    python -m src.data.audit_dataset --data-dir data/raw --output-dir results/dataset_audit
    python -m src.data.audit_dataset --skip-perceptual

``python src/data/audit_dataset.py`` works as well.

Outputs (results/dataset_audit/)
    audit_report.md, dataset_summary.json, class_distribution.csv,
    image_metadata.csv, corrupt_images.csv, exact_duplicates.csv,
    cross_split_duplicates.csv, perceptual_duplicates.csv (optional),
    class_distribution.png, image_dimension_distribution.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import matplotlib

matplotlib.use("Agg")  # headless plotting (no GUI backend required)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    import imagehash  # optional dependency (perceptual duplicates)
except ImportError:
    imagehash = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "dataset_audit"

EXPECTED_SPLITS = ("Training", "Testing")
EXPECTED_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
KAGGLE_DATASET_ID = "masoudnickparvar/brain-tumor-mri-dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

PIL_FORMAT_BY_EXTENSION = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".gif": "GIF",
    ".webp": "WEBP",
}

HASH_CHUNK_SIZE = 1 << 20     # 1 MiB chunks while hashing (keeps RAM flat)
PROGRESS_EVERY = 500          # print progress every N processed files
PHASH_HAMMING_THRESHOLD = 8   # bits (out of 64) - heuristic similarity bound
PHASH_CHUNK_SIZE = 256        # rows per Hamming-distance block
MAX_PHASH_PAIRS = 200_000     # safety cap for near-duplicate pair collection
MAX_LISTED_EXAMPLES = 20      # example rows embedded in the Markdown report

METADATA_COLUMNS = [
    "relative_path", "split", "class", "extension",
    "width", "height", "mode", "channels", "format",
    "file_size_bytes", "sha256", "valid", "error",
]

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class DatasetAuditError(Exception):
    """The dataset is missing or its structure cannot be audited at all."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Return the SHA-256 hex digest of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict[str, Any]:
    """Try to fully decode one image and return its metadata.

    Returns a dict with keys width, height, mode, channels, format, error.
    ``error`` is None for a readable image, otherwise a short description.
    """
    info: dict[str, Any] = {
        "width": None, "height": None, "mode": None,
        "channels": None, "format": None, "error": None,
    }
    try:
        with Image.open(path) as img:
            fmt, mode = img.format, img.mode
            width, height = img.size
            channels = len(img.getbands())
            img.load()  # force a full decode -> detects truncated/corrupt data
        info.update(width=width, height=height, mode=mode, channels=channels, format=fmt)
    except Exception as exc:  # noqa: BLE001 - a single bad file must not stop the audit
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _percentage(part: int, whole: int) -> float:
    """Percentage rounded to 2 decimals (0.0 when the denominator is 0)."""
    return round(part / whole * 100.0, 2) if whole else 0.0


def _sorted_by_reference(names: Iterable[str], reference: tuple[str, ...]) -> list[str]:
    """Sort names with the reference order first, extras afterwards (alphabetical)."""
    name_set = set(names)
    known = [name for name in reference if name in name_set]
    extra = sorted(name for name in name_set if name not in reference)
    return known + extra


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple Markdown table."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1 - directory / class validation
# ---------------------------------------------------------------------------

def discover_dataset(raw_dir: Path) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    """Detect splits and class folders, and collect structural anomalies.

    Returns ``(splits, issues)`` where ``splits`` maps
    ``{split_name: {class_name: class_dir_path}}``.
    Raises :class:`DatasetAuditError` if the dataset or the expected
    Training/Testing layout cannot be found.
    """
    if not raw_dir.is_dir():
        raise DatasetAuditError(
            f"Dataset directory not found: {raw_dir}\n"
            "    The Kaggle 'Brain Tumor MRI Dataset' has not been downloaded yet.\n"
            "    See the 'Dataset' section in README.md for download instructions, then\n"
            "    extract it so that data/raw/Training and data/raw/Testing exist."
        )

    entries = sorted(raw_dir.iterdir())
    root_files = [entry.name for entry in entries if entry.is_file() and entry.name != ".gitkeep"]
    split_dirs = [entry for entry in entries if entry.is_dir()]

    detected_split_names = [entry.name for entry in split_dirs]
    if not any(name in detected_split_names for name in EXPECTED_SPLITS):
        raise DatasetAuditError(
            f"No expected split directories ({', '.join(EXPECTED_SPLITS)}) found in {raw_dir}.\n"
            f"    Found instead: {detected_split_names or 'nothing'}\n"
            "    Expected layout: data/raw/{Training,Testing}/{"
            + ",".join(EXPECTED_CLASSES) + "}/*.jpg"
        )

    splits: dict[str, dict[str, Path]] = {}
    issues: dict[str, Any] = {
        "missing_splits": [name for name in EXPECTED_SPLITS if name not in detected_split_names],
        "missing_classes": {},
        "unexpected_class_dirs": [],
        "unexpected_files_in_split_dirs": [],
        "unexpected_root_files": root_files,
        "nested_dirs_in_class_dirs": [],
    }

    for split_dir in sorted(split_dirs, key=lambda p: _sort_key(p.name, EXPECTED_SPLITS)):
        class_dirs: dict[str, Path] = {}
        for entry in sorted(split_dir.iterdir()):
            if entry.is_dir():
                class_dirs[entry.name] = entry
            else:
                issues["unexpected_files_in_split_dirs"].append(
                    f"{split_dir.name}/{entry.name}"
                )

        missing = [name for name in EXPECTED_CLASSES if name not in class_dirs]
        if missing:
            issues["missing_classes"][split_dir.name] = missing
        issues["unexpected_class_dirs"] += [
            f"{split_dir.name}/{name}" for name in class_dirs
            if name not in EXPECTED_CLASSES
        ]

        for class_name, class_dir in class_dirs.items():
            for nested in sorted(class_dir.iterdir()):
                if nested.is_dir():
                    file_count = sum(1 for p in nested.rglob("*") if p.is_file())
                    issues["nested_dirs_in_class_dirs"].append({
                        "path": f"{split_dir.name}/{class_name}/{nested.name}",
                        "file_count": file_count,
                    })

        splits[split_dir.name] = class_dirs

    return splits, issues


def _sort_key(name: str, reference: tuple[str, ...]) -> tuple[int, str]:
    """Sort key putting reference names first, alphabetically among themselves."""
    if name in reference:
        return (reference.index(name), name)
    return (len(reference), name)


# ---------------------------------------------------------------------------
# Step 2 - per-image audit (integrity, dimensions, hashes)
# ---------------------------------------------------------------------------

def collect_files(splits: dict[str, dict[str, Path]]) -> list[tuple[str, str, Path]]:
    """List every file directly inside the class folders as (split, class, path)."""
    files: list[tuple[str, str, Path]] = []
    split_order = _sorted_by_reference(iter(splits.keys()), EXPECTED_SPLITS)
    for split_name in split_order:
        class_dirs = splits[split_name]
        class_order = _sorted_by_reference(iter(class_dirs.keys()), EXPECTED_CLASSES)
        for class_name in class_order:
            for path in sorted(class_dirs[class_name].iterdir()):
                if path.is_file():
                    files.append((split_name, class_name, path))
    return files


def audit_images(
    raw_dir: Path,
    file_list: list[tuple[str, str, Path]],
    progress_every: int = PROGRESS_EVERY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit every file in the dataset.

    Returns ``(image_rows, non_image_files)``. Image rows follow
    :data:`METADATA_COLUMNS`; non-image files are reported separately.
    """
    image_rows: list[dict[str, Any]] = []
    non_image_files: list[dict[str, Any]] = []
    total = len(file_list)

    for index, (split, class_name, path) in enumerate(file_list, start=1):
        extension = path.suffix.lower()
        relative_path = path.relative_to(raw_dir).as_posix()

        if extension not in IMAGE_EXTENSIONS:
            non_image_files.append({
                "relative_path": relative_path,
                "split": split,
                "class": class_name,
                "extension": extension,
                "file_size_bytes": path.stat().st_size,
            })
            continue

        row: dict[str, Any] = {
            "relative_path": relative_path,
            "split": split,
            "class": class_name,
            "extension": extension,
            "width": None, "height": None, "mode": None, "channels": None,
            "format": None,
            "file_size_bytes": path.stat().st_size,
            "sha256": None,
            "valid": False,
            "error": None,
        }

        info = inspect_image(path)
        if info["error"] is None:
            row.update({key: info[key] for key in
                        ("width", "height", "mode", "channels", "format")})
            try:
                row["sha256"] = sha256_of_file(path)
                row["valid"] = True
            except OSError as exc:
                row["error"] = f"hash failed: {exc}"
        else:
            row["error"] = info["error"]

        image_rows.append(row)

        if index % progress_every == 0 or index == total:
            print(f"    processed {index}/{total} files ...")

    return image_rows, non_image_files


# ---------------------------------------------------------------------------
# Step 3 - exact duplicate detection (SHA-256)
# ---------------------------------------------------------------------------

def find_exact_duplicates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group valid image rows by SHA-256; keep groups with more than one file."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["valid"] and row["sha256"]:
            groups[row["sha256"]].append(row)
    return {digest: members for digest, members in groups.items() if len(members) > 1}


def group_type(members: list[dict[str, Any]]) -> str:
    """Classify a duplicate group as 'same-class' or 'conflicting-class'."""
    return "conflicting-class" if len({m["class"] for m in members}) > 1 else "same-class"


def duplicate_rows(
    duplicate_groups: dict[str, list[dict[str, Any]]],
    only_cross_split: bool = False,
) -> list[dict[str, Any]]:
    """Flatten duplicate groups into CSV rows (one row per file)."""
    rows: list[dict[str, Any]] = []
    ordered = sorted(duplicate_groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for digest, members in ordered:
        crosses = len({m["split"] for m in members}) > 1
        if only_cross_split and not crosses:
            continue
        kind = group_type(members)
        for member in members:
            rows.append({
                "sha256": digest,
                "duplicate_group_size": len(members),
                "group_type": kind,
                "crosses_splits": crosses,
                "relative_path": member["relative_path"],
                "split": member["split"],
                "class": member["class"],
            })
    return rows


def summarise_duplicates(
    duplicate_groups: dict[str, list[dict[str, Any]]],
    valid_images: int,
) -> dict[str, Any]:
    """Compute duplicate / leakage statistics from the exact-duplicate groups."""
    within_by_split: Counter[str] = Counter()
    crossing = 0
    training_files = testing_files = 0
    conflicting_groups = conflicting_files = 0
    affected_files = 0

    for members in duplicate_groups.values():
        split_set = {m["split"] for m in members}
        affected_files += len(members)

        if len({m["class"] for m in members}) > 1:
            conflicting_groups += 1
            conflicting_files += len(members)

        if len(split_set) > 1:
            crossing += 1
            training_files += sum(1 for m in members if m["split"] == "Training")
            testing_files += sum(1 for m in members if m["split"] == "Testing")
        elif split_set:
            within_by_split[next(iter(split_set))] += 1

    return {
        "groups_total": len(duplicate_groups),
        "files_in_duplicate_groups": affected_files,
        "groups_within_splits": dict(within_by_split),
        "groups_crossing_splits": crossing,
        "unique_hashes_crossing_splits": crossing,
        "cross_split_files_training": training_files,
        "cross_split_files_testing": testing_files,
        "conflicting_class_groups": conflicting_groups,
        "conflicting_class_files": conflicting_files,
        "duplicate_file_ratio_percent": _percentage(affected_files, valid_images),
    }


# ---------------------------------------------------------------------------
# Step 4 - optional perceptual (pHash) near-duplicate analysis
# ---------------------------------------------------------------------------

def compute_phash_values(
    raw_dir: Path,
    rows: list[dict[str, Any]],
    progress_every: int = PROGRESS_EVERY,
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    """Compute 64-bit pHash values for valid images.

    Returns ``(hashed_rows, values, failures)`` - images whose hash could not
    be computed are skipped (counted in ``failures``) instead of crashing.
    """
    hashed_rows: list[dict[str, Any]] = []
    values: list[int] = []
    failures = 0
    valid_rows = [row for row in rows if row["valid"]]

    for index, row in enumerate(valid_rows, start=1):
        try:
            with Image.open(raw_dir / row["relative_path"]) as img:
                values.append(int(str(imagehash.phash(img)), 16))
            hashed_rows.append(row)
        except Exception:  # noqa: BLE001 - perceptual stage is best-effort
            failures += 1
        if index % progress_every == 0 or index == len(valid_rows):
            print(f"    perceptual hashing {index}/{len(valid_rows)} images ...")

    return hashed_rows, np.array(values, dtype=np.uint64), failures


def hamming_pairs(
    values: np.ndarray,
    threshold: int,
    chunk_size: int = PHASH_CHUNK_SIZE,
) -> tuple[list[tuple[int, int]], bool]:
    """Find index pairs (i < j) whose hashes differ by at most `threshold` bits."""
    pairs: list[tuple[int, int]] = []
    columns = np.arange(len(values))
    capped = False

    for start in range(0, len(values), chunk_size):
        block = values[start:start + chunk_size]
        xor = block[:, None] ^ values[None, :]
        if hasattr(np, "bitwise_count"):  # numpy >= 2.0
            distances = np.bitwise_count(xor)
        else:
            byte_view = xor.view(np.uint8).reshape(xor.shape + (8,))
            distances = _POPCOUNT_TABLE[byte_view].sum(axis=-1)

        row_indices = np.arange(start, start + block.shape[0])
        mask = (distances <= threshold) & (row_indices[:, None] < columns[None, :])
        pair_rows, pair_cols = np.nonzero(mask)
        for local_row, column in zip(pair_rows.tolist(), pair_cols.tolist()):
            pairs.append((start + local_row, int(column)))
            if len(pairs) >= MAX_PHASH_PAIRS:
                return pairs, True
    return pairs, capped


def group_indices_by_pairs(count: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find: merge connected indices into groups (size >= 2 only)."""
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    touched: set[int] = set()
    for a, b in pairs:
        touched.update((a, b))
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    groups: dict[int, list[int]] = defaultdict(list)
    for node in touched:
        groups[find(node)].append(node)
    return [sorted(members) for members in groups.values() if len(members) > 1]


def run_perceptual_analysis(
    raw_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Best-effort pHash near-duplicate analysis (never breaks the audit)."""
    if imagehash is None:
        return {"status": "skipped", "reason": "imagehash is not installed"}, []

    try:
        hashed_rows, values, failures = compute_phash_values(raw_dir, rows)
        pairs, capped = hamming_pairs(values, PHASH_HAMMING_THRESHOLD)
        components = group_indices_by_pairs(len(values), pairs)
        components.sort(key=len, reverse=True)

        out_rows: list[dict[str, Any]] = []
        cross_split_groups = conflicting_groups = 0
        for group_id, members in enumerate(components, start=1):
            group_rows = [hashed_rows[i] for i in members]
            if len({r["split"] for r in group_rows}) > 1:
                cross_split_groups += 1
            if len({r["class"] for r in group_rows}) > 1:
                conflicting_groups += 1
            for row in group_rows:
                out_rows.append({
                    "group_id": group_id,
                    "group_size": len(group_rows),
                    "relative_path": row["relative_path"],
                    "split": row["split"],
                    "class": row["class"],
                })

        largest_group = max((len(members) for members in components), default=0)
        note = ("Heuristic similarity only - not proof of duplication. "
                "Never delete files based on this section alone.")
        if largest_group > 50:
            note += (" Groups are connected components at the threshold, so very "
                     "large group(s) can reflect transitive similarity chaining "
                     "rather than one duplicate cluster.")
        stats = {
            "status": "completed",
            "method": f"pHash (64-bit), Hamming distance <= {PHASH_HAMMING_THRESHOLD}",
            "images_hashed": len(hashed_rows),
            "hash_failures": failures,
            "near_duplicate_pairs": len(pairs),
            "pairs_capped": capped,
            "groups": len(components),
            "files_in_groups": sum(len(members) for members in components),
            "largest_group_size": largest_group,
            "cross_split_groups": cross_split_groups,
            "conflicting_class_groups": conflicting_groups,
            "note": note,
        }
        return stats, out_rows
    except Exception as exc:  # noqa: BLE001 - audit continues without this stage
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, []


# ---------------------------------------------------------------------------
# Step 5 - summary assembly
# ---------------------------------------------------------------------------

def build_summary(
    raw_dir: Path,
    splits: dict[str, dict[str, Path]],
    issues: dict[str, Any],
    rows: list[dict[str, Any]],
    non_image_files: list[dict[str, Any]],
    duplicate_stats: dict[str, Any],
    perceptual_stats: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the machine-readable dataset_summary.json content."""
    valid_rows = [row for row in rows if row["valid"]]
    invalid_rows = [row for row in rows if not row["valid"]]

    per_split = Counter(row["split"] for row in rows)
    per_split_class: dict[str, dict[str, int]] = {}
    for split_name in _sorted_by_reference(iter(splits.keys()), EXPECTED_SPLITS):
        class_names = _sorted_by_reference(iter(splits[split_name].keys()), EXPECTED_CLASSES)
        per_split_class[split_name] = {
            name: sum(1 for row in rows
                      if row["split"] == split_name and row["class"] == name)
            for name in class_names
        }

    per_class = Counter(row["class"] for row in rows)
    valid_per_class = Counter(row["class"] for row in valid_rows)

    def imbalance(counts: dict[str, int]) -> dict[str, Any]:
        if not counts:
            return {}
        largest = max(counts, key=counts.get)
        smallest = min(counts, key=counts.get)
        return {
            "largest_class": largest,
            "largest_count": counts[largest],
            "smallest_class": smallest,
            "smallest_count": counts[smallest],
            "largest_to_smallest_ratio": round(counts[largest] / counts[smallest], 3)
            if counts[smallest] else None,
        }

    dimension_counts = Counter((row["width"], row["height"]) for row in valid_rows)
    mode_counts = Counter(row["mode"] for row in valid_rows)
    extension_counts = Counter(row["extension"] for row in rows)
    non_image_extension_counts = Counter(entry["extension"] for entry in non_image_files)

    format_mismatches = [
        {"relative_path": row["relative_path"],
         "extension": row["extension"], "actual_format": row["format"]}
        for row in valid_rows
        if PIL_FORMAT_BY_EXTENSION.get(row["extension"]) not in (None, row["format"])
    ]

    total_size_bytes = sum(entry["file_size_bytes"]
                           for entry in rows + non_image_files)

    corrupt_entries = [
        {"relative_path": row["relative_path"], "split": row["split"],
         "class": row["class"], "error": row["error"]}
        for row in invalid_rows
    ]

    warnings: list[str] = []
    if duplicate_stats["groups_crossing_splits"]:
        warnings.append(
            f"{duplicate_stats['groups_crossing_splits']} exact-duplicate group(s) cross "
            f"Training and Testing ({duplicate_stats['cross_split_files_training']} Training + "
            f"{duplicate_stats['cross_split_files_testing']} Testing files) - potential leakage."
        )
    if duplicate_stats["conflicting_class_groups"]:
        warnings.append(
            f"{duplicate_stats['conflicting_class_groups']} duplicate group(s) contain "
            "conflicting class labels for identical content."
        )
    if invalid_rows:
        warnings.append(f"{len(invalid_rows)} corrupt/unreadable image file(s) found.")
    if non_image_files:
        warnings.append(f"{len(non_image_files)} non-image file(s) found inside class folders.")
    if format_mismatches:
        warnings.append(
            f"{len(format_mismatches)} image(s) whose real format does not match their extension."
        )
    for split_name, missing in issues["missing_classes"].items():
        warnings.append(f"Missing expected class folder(s) in {split_name}: {', '.join(missing)}.")
    if issues["missing_splits"]:
        warnings.append(f"Missing expected split folder(s): {', '.join(issues['missing_splits'])}.")
    if issues["nested_dirs_in_class_dirs"]:
        warnings.append("Unexpected nested director(ies) found inside class folders "
                        "(their files are NOT counted as dataset images).")

    try:
        dataset_root = raw_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        dataset_root = str(raw_dir)

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "audit_script": "src/data/audit_dataset.py",
        "dataset": {
            "name": "Brain Tumor MRI Dataset",
            "kaggle_id": KAGGLE_DATASET_ID,
            "dataset_root": dataset_root,
            "expected_splits": list(EXPECTED_SPLITS),
            "expected_classes": list(EXPECTED_CLASSES),
            "detected_splits": _sorted_by_reference(iter(splits.keys()), EXPECTED_SPLITS),
            "detected_classes": _sorted_by_reference(
                {name for class_dirs in splits.values() for name in class_dirs},
                EXPECTED_CLASSES,
            ),
        },
        "structure_issues": {
            "missing_splits": issues["missing_splits"],
            "missing_classes": issues["missing_classes"],
            "unexpected_class_dirs": issues["unexpected_class_dirs"],
            "unexpected_files_in_split_dirs": issues["unexpected_files_in_split_dirs"],
            "unexpected_root_files": issues["unexpected_root_files"],
            "nested_dirs_in_class_dirs": issues["nested_dirs_in_class_dirs"],
            "non_image_files": non_image_files,
        },
        "counts": {
            "total_files_scanned": len(rows) + len(non_image_files),
            "total_images": len(rows),
            "valid_images": len(valid_rows),
            "invalid_images": len(invalid_rows),
            "non_image_files": len(non_image_files),
            "per_split": dict(per_split),
            "per_split_class": per_split_class,
            "per_class": dict(per_class),
            "valid_per_class": dict(valid_per_class),
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        },
        "imbalance": {
            "dataset_level": imbalance(dict(per_class)),
            "per_split": {
                split_name: imbalance(class_counts)
                for split_name, class_counts in per_split_class.items()
            },
        },
        "extensions": {
            "image_extensions": dict(extension_counts.most_common()),
            "non_image_extensions": dict(non_image_extension_counts.most_common()),
        },
        "dimensions": {
            "unique_combinations": len(dimension_counts),
            "min_width": min((w for w, _ in dimension_counts), default=None),
            "max_width": max((w for w, _ in dimension_counts), default=None),
            "min_height": min((h for _, h in dimension_counts), default=None),
            "max_height": max((h for _, h in dimension_counts), default=None),
            "most_common": [
                {"width": w, "height": h, "count": count,
                 "percentage": _percentage(count, len(valid_rows))}
                for (w, h), count in dimension_counts.most_common(10)
            ],
        },
        "modes": {
            "counts_by_mode": dict(mode_counts.most_common()),
            "grayscale_L": mode_counts.get("L", 0),
            "rgb": mode_counts.get("RGB", 0),
            "rgba": mode_counts.get("RGBA", 0),
            "unusual_modes": {m: c for m, c in mode_counts.items()
                              if m not in ("L", "RGB", "RGBA")},
        },
        "format_extension_mismatches": {
            "count": len(format_mismatches),
            "examples": format_mismatches[:MAX_LISTED_EXAMPLES],
        },
        "corrupt_images": {
            "count": len(corrupt_entries),
            "by_split": dict(Counter(entry["split"] for entry in corrupt_entries)),
            "examples": corrupt_entries[:MAX_LISTED_EXAMPLES],
        },
        "duplicates": duplicate_stats,
        "perceptual_duplicates": perceptual_stats,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Step 6 - outputs (CSV / JSON / PNG / Markdown)
# ---------------------------------------------------------------------------

def write_csv(path: Path, columns: list[str], records: list[dict[str, Any]]) -> None:
    """Write records to CSV with a stable column order (header-only if empty)."""
    pd.DataFrame(records, columns=columns).to_csv(path, index=False, encoding="utf-8")


def write_class_distribution(path: Path, summary: dict[str, Any]) -> None:
    """class_distribution.csv: split x class counts and percentages."""
    records = []
    for split_name, class_counts in summary["counts"]["per_split_class"].items():
        split_total = sum(class_counts.values())
        for class_name, count in class_counts.items():
            records.append({
                "split": split_name,
                "class": class_name,
                "image_count": count,
                "percentage": _percentage(count, split_total),
                "percentage_of_total": _percentage(
                    count, summary["counts"]["total_images"]
                ),
            })
    write_csv(path, ["split", "class", "image_count", "percentage",
                     "percentage_of_total"], records)


def plot_class_distribution(summary: dict[str, Any], out_path: Path) -> None:
    """Grouped bar chart: image count per class, separated by split."""
    per_split_class = summary["counts"]["per_split_class"]
    splits = list(per_split_class.keys())
    classes = _sorted_by_reference(
        {name for class_counts in per_split_class.values() for name in class_counts},
        EXPECTED_CLASSES,
    )
    if not splits or not classes:
        return

    positions = np.arange(len(classes))
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, split_name in enumerate(splits):
        counts = [per_split_class[split_name].get(name, 0) for name in classes]
        bars = ax.bar(positions + offset * width, counts, width, label=split_name)
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(positions + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylabel("number of images")
    ax.set_title("Image count per class and split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dimension_distribution(rows: list[dict[str, Any]], out_path: Path) -> int:
    """Bar chart of the most common width x height combinations. Returns unique count."""
    dimension_counts = Counter((row["width"], row["height"])
                               for row in rows if row["valid"])
    if not dimension_counts:
        return 0
    top = dimension_counts.most_common(15)
    labels = [f"{w}x{h}" for (w, h), _ in top]
    counts = [count for _, count in top]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts)
    ax.bar_label(bars, fontsize=8, padding=2)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("number of images")
    ax.set_xlabel(f"unique dimension combinations: {len(dimension_counts)}")
    ax.set_title("Most common image dimensions (top 15)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return len(dimension_counts)


def build_markdown_report(summary: dict[str, Any]) -> str:
    """Render the human-readable audit_report.md content."""
    dataset = summary["dataset"]
    counts = summary["counts"]
    duplicates = summary["duplicates"]
    perceptual = summary["perceptual_duplicates"]
    rows: list[str] = []

    rows.append("# Dataset Audit Report")
    rows.append("")
    rows.append(f"_Generated: {summary['generated_at_utc']} - Phase 2 "
                "(read-only audit; no image was modified, resized or deleted)._")
    rows.append("")

    rows.append("## Dataset")
    rows.append(f"- Name: Brain Tumor MRI Dataset")
    rows.append(f"- Kaggle identifier: `{dataset['kaggle_id']}` (public dataset, CC BY 4.0)")
    rows.append(f"- Source: downloaded with the Kaggle CLI; not collected by this project")
    rows.append(f"- Local location: `{dataset['dataset_root']}` (excluded from Git)")
    rows.append(f"- Audit script: `python -m src.data.audit_dataset`")
    rows.append("")

    rows.append("## Dataset structure")
    rows.append(f"- Detected splits: {', '.join(dataset['detected_splits']) or 'none'}")
    rows.append(f"- Detected classes: {', '.join(dataset['detected_classes']) or 'none'}")
    issues = summary["structure_issues"]
    anomalies = []
    if issues["missing_splits"]:
        anomalies.append(f"missing split folder(s): {', '.join(issues['missing_splits'])}")
    for split_name, missing in issues["missing_classes"].items():
        anomalies.append(f"missing class folder(s) in {split_name}: {', '.join(missing)}")
    if issues["unexpected_class_dirs"]:
        anomalies.append(f"unexpected class folder(s): {', '.join(issues['unexpected_class_dirs'])}")
    if issues["unexpected_files_in_split_dirs"]:
        anomalies.append(f"unexpected file(s) directly in split folders: "
                         f"{', '.join(issues['unexpected_files_in_split_dirs'])}")
    if issues["unexpected_root_files"]:
        anomalies.append(f"unexpected file(s) in dataset root: "
                         f"{', '.join(issues['unexpected_root_files'])}")
    if issues["nested_dirs_in_class_dirs"]:
        nested = ", ".join(f"{d['path']} ({d['file_count']} files)"
                           for d in issues["nested_dirs_in_class_dirs"])
        anomalies.append(f"unexpected nested director(ies) inside class folders "
                         f"(files NOT counted): {nested}")
    rows.append(f"- Structural anomalies: {'; '.join(anomalies) if anomalies else 'none detected'}")
    rows.append("")

    rows.append("## Total images")
    rows.append(f"- Total image files: **{counts['total_images']:,}**")
    rows.append(f"- Valid images: **{counts['valid_images']:,}**")
    rows.append(f"- Invalid / unreadable images: **{counts['invalid_images']:,}**")
    rows.append(f"- Non-image files inside class folders: **{counts['non_image_files']:,}**")
    rows.append(f"- Total dataset size: {counts['total_size_mb']:,} MB")
    rows.append("")

    rows.append("## Class distribution")
    distribution_rows = []
    for split_name, class_counts in counts["per_split_class"].items():
        split_total = sum(class_counts.values())
        for class_name, count in class_counts.items():
            distribution_rows.append([
                split_name, class_name, f"{count:,}",
                f"{_percentage(count, split_total):.2f}%",
                f"{_percentage(count, counts['total_images']):.2f}%",
            ])
    rows.append(_md_table(["split", "class", "count", "% of split", "% of dataset"],
                          distribution_rows))
    dataset_imbalance = summary["imbalance"]["dataset_level"]
    if dataset_imbalance:
        rows.append("")
        rows.append(f"- Largest class (dataset-wide): {dataset_imbalance['largest_class']} "
                    f"({dataset_imbalance['largest_count']:,})")
        rows.append(f"- Smallest class (dataset-wide): {dataset_imbalance['smallest_class']} "
                    f"({dataset_imbalance['smallest_count']:,})")
        rows.append(f"- Largest-to-smallest ratio: {dataset_imbalance['largest_to_smallest_ratio']}")
    rows.append("")

    rows.append("## Train/Test distribution")
    split_rows = [[split_name, f"{count:,}",
                   f"{_percentage(count, counts['total_images']):.2f}%"]
                  for split_name, count in counts["per_split"].items()]
    rows.append(_md_table(["split", "count", "% of dataset"], split_rows))
    rows.append("")

    rows.append("## File formats")
    extension_rows = [[ext or "(no extension)", f"{n:,}"]
                      for ext, n in summary["extensions"]["image_extensions"].items()]
    rows.append(_md_table(["image extension", "count"], extension_rows))
    if summary["extensions"]["non_image_extensions"]:
        rows.append("")
        rows.append("Unexpected non-image extensions found inside class folders: "
                    + ", ".join(f"`{ext or '(none)'}` ({n})"
                                for ext, n in summary["extensions"]["non_image_extensions"].items()))
        for entry in issues["non_image_files"][:MAX_LISTED_EXAMPLES]:
            rows.append(f"- `{entry['relative_path']}`")
    mismatch = summary["format_extension_mismatches"]
    if mismatch["count"]:
        rows.append("")
        rows.append(f"Format/extension mismatches (file content vs filename): "
                    f"**{mismatch['count']}** - see `dataset_summary.json` for examples.")
    rows.append("")

    rows.append("## Image dimensions")
    dims = summary["dimensions"]
    rows.append(f"- Width range: {dims['min_width']} - {dims['max_width']} px")
    rows.append(f"- Height range: {dims['min_height']} - {dims['max_height']} px")
    rows.append(f"- Unique dimension combinations: **{dims['unique_combinations']}**")
    if dims["most_common"]:
        rows.append("")
        rows.append(_md_table(
            ["width", "height", "count", "% of valid images"],
            [[d["width"], d["height"], f"{d['count']:,}", f"{d['percentage']:.2f}%"]
             for d in dims["most_common"]],
        ))
    rows.append("")
    rows.append("No resizing was performed during this audit "
                "(a target size decision belongs to Phase 3).")
    rows.append("")

    rows.append("## Image modes/channels")
    mode_rows = [[mode, f"{n:,}"] for mode, n in summary["modes"]["counts_by_mode"].items()]
    rows.append(_md_table(["PIL mode", "count"], mode_rows))
    rows.append("")
    rows.append(f"- Grayscale (`L`): {summary['modes']['grayscale_L']:,} | "
                f"RGB: {summary['modes']['rgb']:,} | RGBA: {summary['modes']['rgba']:,}"
                + (f" | unusual modes: {summary['modes']['unusual_modes']}"
                   if summary["modes"]["unusual_modes"] else ""))
    rows.append("")
    rows.append("Note: an MRI can visually look grayscale while being stored as RGB; "
                "the mode above reflects the stored pixel format, not the visual impression.")
    rows.append("")

    rows.append("## Corrupt files")
    if summary["corrupt_images"]["count"] == 0:
        rows.append("No corrupt or unreadable images were detected.")
    else:
        rows.append(f"**{summary['corrupt_images']['count']}** corrupt/unreadable file(s) found "
                    f"({summary['corrupt_images']['by_split']}). Full list in `corrupt_images.csv`.")
        rows.append("")
        rows.append(_md_table(
            ["relative path", "split", "class", "error"],
            [[entry["relative_path"], entry["split"], entry["class"], entry["error"]]
             for entry in summary["corrupt_images"]["examples"]],
        ))
    rows.append("")

    rows.append("## Exact duplicates")
    if duplicates["groups_total"] == 0:
        rows.append("No exact duplicates (identical SHA-256 content) were detected.")
    else:
        rows.append(f"- Duplicate groups: **{duplicates['groups_total']:,}**")
        rows.append(f"- Files inside duplicate groups: **{duplicates['files_in_duplicate_groups']:,}** "
                    f"({duplicates['duplicate_file_ratio_percent']:.2f}% of valid images)")
        rows.append(f"- Groups fully inside one split: {duplicates['groups_within_splits']}")
        rows.append(f"- Groups crossing splits: **{duplicates['groups_crossing_splits']:,}**")
        rows.append(f"- Groups with conflicting class labels for identical content: "
                    f"**{duplicates['conflicting_class_groups']:,}**")
        rows.append("")
        rows.append("Full listing: `exact_duplicates.csv` "
                    "(and `cross_split_duplicates.csv` for leakage-relevant groups).")
    rows.append("")

    rows.append("## Cross-split duplicates")
    if duplicates["groups_crossing_splits"] == 0:
        rows.append("No exact-duplicate group overlaps between Training and Testing "
                    "was detected by SHA-256 hashing.")
    else:
        rows.append(f"- Unique duplicate hashes crossing Training/Testing: "
                    f"**{duplicates['unique_hashes_crossing_splits']:,}**")
        rows.append(f"- Affected Training files: **{duplicates['cross_split_files_training']:,}**")
        rows.append(f"- Affected Testing files: **{duplicates['cross_split_files_testing']:,}**")
        if duplicates["conflicting_class_files"]:
            rows.append(f"- WARNING: {duplicates['conflicting_class_groups']} group(s) carry "
                        f"different labels for identical content "
                        f"({duplicates['conflicting_class_files']} files).")
        rows.append("")
        rows.append("See `cross_split_duplicates.csv` for the exact file list.")
    rows.append("")

    rows.append("## Perceptual near-duplicates (heuristic)")
    if perceptual.get("status") != "completed":
        rows.append(f"Perceptual analysis not completed (status: "
                    f"{perceptual.get('status')}; {perceptual.get('reason') or perceptual.get('error') or ''}).")
    else:
        rows.append(f"- Method: {perceptual['method']}")
        rows.append(f"- Images hashed: {perceptual['images_hashed']:,} "
                    f"(failures: {perceptual['hash_failures']})")
        rows.append(f"- Near-duplicate group(s): **{perceptual['groups']:,}** "
                    f"covering {perceptual['files_in_groups']:,} files "
                    f"(largest group: {perceptual['largest_group_size']:,})")
        rows.append(f"- Groups crossing splits: {perceptual['cross_split_groups']:,}; "
                    f"conflicting-label groups: {perceptual['conflicting_class_groups']:,}")
        rows.append("")
        rows.append(f"{perceptual['note']} See `perceptual_duplicates.csv` "
                    "(reported separately from exact SHA-256 duplicates).")
    rows.append("")

    rows.append("## Potential leakage concerns")
    if duplicates["groups_crossing_splits"] == 0 and duplicates["groups_total"] == 0:
        rows.append("The audit did not detect exact-content overlap between Training and Testing.")
    elif duplicates["groups_crossing_splits"] == 0:
        rows.append(f"Duplicates exist inside splits ({duplicates['groups_total']} group(s)) "
                    "but no exact-content overlap between Training and Testing was detected.")
    else:
        rows.append(f"Identical file content appears in both Training and Testing "
                    f"({duplicates['groups_crossing_splits']} group(s); "
                    f"{duplicates['cross_split_files_training']} Training and "
                    f"{duplicates['cross_split_files_testing']} Testing files). "
                    "This is a train/test leakage risk for model evaluation and must be "
                    "resolved before trusting the existing split.")
    rows.append("")

    rows.append("## Observations")
    if summary["warnings"]:
        rows += [f"- {warning}" for warning in summary["warnings"]]
    else:
        rows.append("- No data-quality warnings were triggered by this audit.")
    rows.append("")

    rows.append("## Decision required before Phase 3")
    rows.append("- Decide how to handle cross-split exact duplicates "
                "(drop duplicates, relabel, or rebuild the train/test split).")
    rows.append("- Decide class-imbalance handling (report only - see distribution above).")
    rows.append("- Decide the input image size (224x224 is the current plan) and "
                "colour handling (grayscale vs RGB) for preprocessing.")
    rows.append("- Decide the policy for any corrupt / unusual files listed above.")
    rows.append("- Decide whether perceptual findings need manual review.")
    rows.append("")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the full audit. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Read-only audit of the Brain Tumor MRI dataset in data/raw.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="dataset root (default: data/raw)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="report directory (default: results/dataset_audit)")
    parser.add_argument("--skip-perceptual", action="store_true",
                        help="skip the optional pHash near-duplicate analysis")
    args = parser.parse_args(argv)

    raw_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("=" * 60)
    print(" Brain Tumor Dataset Audit")
    print("=" * 60)
    print(f"Dataset root: {raw_dir}")
    print(f"Output dir  : {output_dir}")
    print()

    try:
        splits, issues = discover_dataset(raw_dir)
    except DatasetAuditError as exc:
        print(f"[ERROR] {exc}")
        return 1

    detected_splits = _sorted_by_reference(iter(splits.keys()), EXPECTED_SPLITS)
    detected_classes = _sorted_by_reference(
        {name for class_dirs in splits.values() for name in class_dirs}, EXPECTED_CLASSES)
    print(f"Detected splits : {', '.join(detected_splits)}")
    print(f"Detected classes: {', '.join(detected_classes)}")

    structural_problems = []
    if issues["missing_splits"]:
        structural_problems.append(f"missing split(s): {', '.join(issues['missing_splits'])}")
    for split_name, missing in issues["missing_classes"].items():
        structural_problems.append(f"missing class(es) in {split_name}: {', '.join(missing)}")
    if issues["unexpected_class_dirs"]:
        structural_problems.append(f"unexpected class folder(s): {', '.join(issues['unexpected_class_dirs'])}")
    if issues["unexpected_files_in_split_dirs"]:
        structural_problems.append(f"unexpected file(s) in split folders: {', '.join(issues['unexpected_files_in_split_dirs'])}")
    if issues["unexpected_root_files"]:
        structural_problems.append(f"unexpected file(s) in dataset root: {', '.join(issues['unexpected_root_files'])}")
    if issues["nested_dirs_in_class_dirs"]:
        structural_problems.append(f"{len(issues['nested_dirs_in_class_dirs'])} nested director(ies) inside class folders")
    if structural_problems:
        print("\nStructure warnings:")
        for problem in structural_problems:
            print(f"  - {problem}")
    print()

    file_list = collect_files(splits)
    print(f"Scanning images ({len(file_list)} files)...")
    rows, non_image_files = audit_images(raw_dir, file_list)

    print("\nPer split/class (image files):")
    for split_name in detected_splits:
        print(f"  {split_name}:")
        for class_name in _sorted_by_reference(iter(splits[split_name].keys()), EXPECTED_CLASSES):
            class_rows = [row for row in rows
                          if row["split"] == split_name and row["class"] == class_name]
            invalid = sum(1 for row in class_rows if not row["valid"])
            suffix = f" ({invalid} invalid)" if invalid else ""
            print(f"    {class_name}: {len(class_rows)} images{suffix}")

    print("\nDetecting exact duplicates (SHA-256)...")
    duplicate_groups = find_exact_duplicates(rows)
    duplicate_stats = summarise_duplicates(duplicate_groups, sum(1 for r in rows if r["valid"]))
    print(f"  {duplicate_stats['groups_total']} duplicate group(s), "
          f"{duplicate_stats['groups_crossing_splits']} crossing Training/Testing")

    if args.skip_perceptual:
        perceptual_stats, perceptual_rows = {"status": "skipped", "reason": "disabled by user"}, []
        print("\nPerceptual analysis skipped (--skip-perceptual).")
    elif imagehash is None:
        perceptual_stats, perceptual_rows = {"status": "skipped", "reason": "imagehash not installed"}, []
        print("\nPerceptual analysis skipped (imagehash not installed).")
    else:
        print("\nRunning perceptual near-duplicate analysis (pHash)...")
        perceptual_stats, perceptual_rows = run_perceptual_analysis(raw_dir, rows)
        print(f"  status: {perceptual_stats['status']}"
              + (f", {perceptual_stats['groups']} near-duplicate group(s)"
                 if perceptual_stats["status"] == "completed" else ""))

    summary = build_summary(
        raw_dir, splits, issues, rows, non_image_files, duplicate_stats, perceptual_stats)

    print("\nWriting reports...")
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "image_metadata.csv", METADATA_COLUMNS, rows)
    write_csv(output_dir / "corrupt_images.csv",
              ["relative_path", "split", "class", "error"],
              [{"relative_path": row["relative_path"], "split": row["split"],
                "class": row["class"], "error": row["error"]}
               for row in rows if not row["valid"]])
    write_csv(output_dir / "exact_duplicates.csv",
              ["sha256", "duplicate_group_size", "group_type", "crosses_splits",
               "relative_path", "split", "class"],
              duplicate_rows(duplicate_groups))
    write_csv(output_dir / "cross_split_duplicates.csv",
              ["sha256", "duplicate_group_size", "group_type", "crosses_splits",
               "relative_path", "split", "class"],
              duplicate_rows(duplicate_groups, only_cross_split=True))
    if perceptual_rows:
        write_csv(output_dir / "perceptual_duplicates.csv",
                  ["group_id", "group_size", "relative_path", "split", "class"],
                  perceptual_rows)
    write_class_distribution(output_dir / "class_distribution.csv", summary)

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("Generating plots...")
    plot_class_distribution(summary, output_dir / "class_distribution.png")
    plot_dimension_distribution(rows, output_dir / "image_dimension_distribution.png")

    with (output_dir / "audit_report.md").open("w", encoding="utf-8") as handle:
        handle.write(build_markdown_report(summary) + "\n")

    print()
    print("=" * 60)
    print(" Audit completed.")
    print(f" Total images        : {summary['counts']['total_images']} "
          f"(valid: {summary['counts']['valid_images']}, "
          f"invalid: {summary['counts']['invalid_images']})")
    print(f" Non-image files     : {summary['counts']['non_image_files']}")
    print(f" Exact duplicates    : {duplicate_stats['groups_total']} group(s), "
          f"{duplicate_stats['files_in_duplicate_groups']} file(s) "
          f"(cross-split: {duplicate_stats['groups_crossing_splits']})")
    if summary["warnings"]:
        print("\n Warnings:")
        for warning in summary["warnings"]:
            print(f"   - {warning}")
    print()
    print(f" Report: {(output_dir / 'audit_report.md')}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
