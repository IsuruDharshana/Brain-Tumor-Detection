"""
evaluate_models.py
------------------
Phase 6 final evaluation script for the Brain Tumor MRI Classification
project.

This script evaluates two pre-trained models on the official held-out test
set (data/processed/test_manifest.csv) and produces reproducible artefacts
for Phase 7 and the project report.

OFFICIAL TEST-SET POLICY
-------------------------
The test set has been untouched throughout Phases 1–5.
It must be evaluated EXACTLY ONCE.

A mandatory command-line flag guards against accidental invocation:

    python -m src.evaluation.evaluate_models --confirm-test-evaluation

Running WITHOUT this flag prints a warning and exits immediately.

MODELS EVALUATED
----------------
1. Baseline custom CNN       — models/baseline_cnn.keras
2. EfficientNetB0 transfer   — models/efficientnet_b0.keras

PREPROCESSING
-------------
Both saved models accept Phase 3 pipeline output: float32, [0,1], 224×224×3.

Baseline CNN:   accepts [0,1] directly.
EfficientNetB0: the saved model contains a Rescaling(255.0) adapter layer;
                inputs are passed as [0,1] and the model rescales internally.

DO NOT call efficientnet.preprocess_input().
DO NOT multiply by 255 outside the model.

OUTPUTS (written to results/final_evaluation/)
----------------------------------------------
    baseline_metrics.json
    efficientnet_metrics.json
    baseline_classification_report.csv
    efficientnet_classification_report.csv
    baseline_predictions.csv
    efficientnet_predictions.csv
    baseline_confusion_matrix.csv
    efficientnet_confusion_matrix.csv
    baseline_confusion_matrix.png
    efficientnet_confusion_matrix.png
    baseline_confusion_matrix_normalized.png
    efficientnet_confusion_matrix_normalized.png
    model_comparison.csv
    model_comparison.json
    duplicate_sensitivity/
        baseline_metrics.json
        efficientnet_metrics.json
        comparison.csv
    evaluation_summary.json

Usage:
    python -m src.evaluation.evaluate_models --confirm-test-evaluation

EDUCATIONAL / RESEARCH SOFTWARE DISCLAIMER
-------------------------------------------
This project classifies brain MRI images for educational and research
purposes only.  It is NOT a clinical diagnostic tool.  Results must NOT
be used for medical decision-making.  Grad-CAM visualisations are
explanatory only and do NOT constitute tumour segmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    ID_TO_LABEL,
    LABEL_TO_ID,
    RANDOM_SEED,
    REPO_ROOT,
    TEST_MANIFEST,
)
from src.data.data_pipeline import build_dataset, load_manifest
from src.evaluation.metrics import (
    build_comparison_df,
    compute_classification_report_df,
    compute_confusion_matrix,
    compute_metrics,
    normalize_confusion_matrix,
    select_winner,
)
from src.explainability.gradcam import find_conv_layer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR: Path = REPO_ROOT / "results" / "final_evaluation"
MODELS_DIR: Path = REPO_ROOT / "models"
DUP_RESULTS_DIR: Path = RESULTS_DIR / "duplicate_sensitivity"
GRADCAM_DIR: Path = RESULTS_DIR / "gradcam"

TEST_DUPS_CSV: Path = (
    REPO_ROOT / "results" / "preprocessing" / "test_duplicates.csv"
)

MODEL_CONFIGS: list[dict] = [
    {
        "key": "baseline",
        "display_name": "Baseline CNN",
        "path": MODELS_DIR / "baseline_cnn.keras",
        "preprocessing": (
            "Phase 3 pipeline output [0,1] float32 passed directly. "
            "No external rescaling."
        ),
        "gradcam_preferred_layer": "block4_conv",
    },
    {
        "key": "efficientnet",
        "display_name": "EfficientNetB0 Transfer",
        "path": MODELS_DIR / "efficientnet_b0.keras",
        "preprocessing": (
            "Phase 3 pipeline output [0,1] float32 passed directly. "
            "Model contains internal Rescaling(255.0) adapter — "
            "DO NOT rescale externally."
        ),
        "gradcam_preferred_layer": None,  # auto-detect last Conv2D in backbone
    },
]

BATCH_SIZE: int = DEFAULT_BATCH_SIZE
PRIMARY_METRIC: str = "macro_f1"

DISCLAIMER = (
    "EDUCATIONAL/RESEARCH SOFTWARE. "
    "This project classifies brain MRI images for educational and research "
    "purposes only. It is NOT a clinical diagnostic tool. "
    "Results must NOT be used for medical decision-making. "
    "Grad-CAM visualisations are explanatory only and do NOT constitute "
    "tumour segmentation or proof of lesion location."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guard_flag() -> None:
    """Parse argv and exit unless --confirm-test-evaluation is present."""
    parser = argparse.ArgumentParser(
        description="Phase 6 final test-set evaluation."
    )
    parser.add_argument(
        "--confirm-test-evaluation",
        action="store_true",
        help=(
            "Required flag. Confirms intentional access to the untouched "
            "official test set. Without this flag the script exits immediately."
        ),
    )
    args = parser.parse_args()
    if not args.confirm_test_evaluation:
        print()
        print("=" * 70)
        print("PHASE 6 FINAL TEST EVALUATION")
        print("=" * 70)
        print()
        print("  This script accesses the UNTOUCHED OFFICIAL TEST SET.")
        print("  The test set should be evaluated exactly once.")
        print()
        print("  To proceed, run:")
        print()
        print(
            "    python -m src.evaluation.evaluate_models "
            "--confirm-test-evaluation"
        )
        print()
        print("  Exiting without loading the test set.")
        print("=" * 70)
        sys.exit(0)


def _load_duplicate_paths(dup_csv: Path) -> set[str]:
    """Return the set of relative_path values flagged is_representative=False.

    These are the REDUNDANT copies to remove for the sensitivity analysis.
    The 'relative_path' column stores repo-root-relative POSIX paths matching
    the manifest format (e.g. 'data/raw/Testing/glioma/Te-gl_303.jpg').
    """
    redundant: set[str] = set()
    if not dup_csv.is_file():
        return redundant
    with dup_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            is_rep = row.get("is_representative", "").strip().lower()
            if is_rep in ("false", "0", "no"):
                rel = row.get("relative_path", "").strip()
                if rel:
                    # Normalise to forward slashes for matching.
                    redundant.add(rel.replace("\\", "/"))
    return redundant


def _build_predictions_df(
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """Assemble a structured DataFrame from one inference pass.

    Args:
        paths:  absolute file paths in the same order as the dataset.
        y_true: int32 array (N,) of ground-truth labels.
        y_pred: int32 array (N,) of predicted class indices.
        y_prob: float32 array (N, num_classes) of class probabilities.

    Returns:
        pd.DataFrame with columns:
            filepath, true_label_id, true_label_name,
            predicted_label_id, predicted_label_name, confidence,
            prob_glioma, prob_meningioma, prob_notumor, prob_pituitary
    """
    num_classes = y_prob.shape[1]
    rows: list[dict] = []
    for i, path in enumerate(paths):
        pred_id = int(y_pred[i])
        true_id = int(y_true[i])
        row: dict = {
            "filepath": path,
            "true_label_id": true_id,
            "true_label_name": ID_TO_LABEL.get(true_id, str(true_id)),
            "predicted_label_id": pred_id,
            "predicted_label_name": ID_TO_LABEL.get(pred_id, str(pred_id)),
            "confidence": float(y_prob[i, pred_id]),
        }
        for j in range(num_classes):
            row[f"prob_{CLASS_NAMES[j]}"] = float(y_prob[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def _filter_predictions_df(
    pred_df: pd.DataFrame,
    redundant_paths: set[str],
    repo_root: Path,
) -> pd.DataFrame:
    """Return a copy of pred_df with redundant duplicate rows removed.

    Matching uses repo-relative POSIX paths so that absolute paths stored
    in the 'filepath' column can be compared against the relative paths in
    ``redundant_paths`` (read from test_duplicates.csv).

    The original DataFrame is NEVER mutated.

    Args:
        pred_df:        full-test predictions DataFrame from _build_predictions_df.
        redundant_paths: set of repo-relative paths to exclude
                        (is_representative=False rows from test_duplicates.csv).
        repo_root:      repository root used to strip the absolute prefix.

    Returns:
        A new DataFrame containing only non-redundant rows.
    """
    def _to_rel(abs_path: str) -> str:
        try:
            return str(Path(abs_path).relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            # Path is not under repo_root — keep as-is, won't match any rel path.
            return str(abs_path).replace("\\", "/")

    mask = ~pred_df["filepath"].apply(_to_rel).isin(redundant_paths)
    return pred_df[mask].copy()



def _run_model_prediction(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a dataset and return (y_true, y_pred, y_prob).

    Args:
        model: loaded Keras model.
        dataset: tf.data.Dataset yielding (images, labels) batches.

    Returns:
        y_true:  int32 array (N,) of ground-truth labels.
        y_pred:  int32 array (N,) of predicted class indices.
        y_prob:  float32 array (N, num_classes) of class probabilities.
    """
    all_true: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        all_true.append(labels.numpy())
        all_prob.append(probs)
    y_true = np.concatenate(all_true, axis=0).astype(np.int32)
    y_prob = np.concatenate(all_prob, axis=0).astype(np.float32)
    y_pred = y_prob.argmax(axis=1).astype(np.int32)
    return y_true, y_pred, y_prob


def _save_predictions_df(
    pred_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write the predictions DataFrame to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)


def _compute_sensitivity_from_predictions(
    pred_df: pd.DataFrame,
    redundant_paths: set[str],
    display_name: str,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Recompute metrics by filtering redundant duplicate rows from pred_df.

    This is the SECONDARY duplicate-sensitivity analysis.
    It does NOT call model.predict() or load a dataset.
    It operates exclusively on the already-computed prediction DataFrame.

    The original pred_df is NOT mutated (filtering uses a copy).

    Args:
        pred_df:         full-test predictions DataFrame (all 1600 rows).
        redundant_paths: set of repo-relative paths that are redundant
                         duplicates (is_representative=False).
        display_name:    human-readable model name for the result dict.
        repo_root:       used to convert absolute filepaths to relative.

    Returns:
        dict with the same keys as compute_metrics() plus 'model', 'dataset',
        and 'note' metadata fields.
    """
    DUP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not redundant_paths:
        result: dict = {
            "note": (
                "No redundant duplicates found; sensitivity analysis skipped."
            )
        }
        return result

    dedup_df = _filter_predictions_df(pred_df, redundant_paths, repo_root)
    removed = len(pred_df) - len(dedup_df)

    print(
        f"  Sensitivity: {len(pred_df)} → {len(dedup_df)} predictions "
        f"after filtering {removed} redundant duplicate rows. "
        f"No additional inference performed."
    )

    y_true_d = dedup_df["true_label_id"].values.astype(np.int32)
    y_pred_d = dedup_df["predicted_label_id"].values.astype(np.int32)

    dedup_metrics = compute_metrics(y_true_d, y_pred_d, class_names=CLASS_NAMES)
    dedup_metrics["model"] = display_name
    dedup_metrics["dataset"] = (
        f"deduplicated test set ({len(dedup_df)} images, "
        f"{removed} redundant copies filtered from primary predictions)"
    )
    dedup_metrics["note"] = (
        "SECONDARY SENSITIVITY ANALYSIS — NOT the official test score. "
        "Primary metrics use the full 1600-image official test set. "
        "No additional model inference was performed for this analysis."
    )
    return dedup_metrics


def _save_confusion_matrix_png(
    cm: np.ndarray,
    title: str,
    output_path: Path,
    normalized: bool = False,
) -> None:
    """Save a confusion matrix as a labelled PNG."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest",
                   cmap="Blues" if not normalized else "Blues",
                   vmin=0, vmax=(1.0 if normalized else None))
    fig.colorbar(im, ax=ax)
    ax.set_title(title, pad=14)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    tick_marks = range(len(CLASS_NAMES))
    ax.set_xticks(list(tick_marks))
    ax.set_yticks(list(tick_marks))
    ax.set_xticklabels(list(CLASS_NAMES), rotation=30, ha="right")
    ax.set_yticklabels(list(CLASS_NAMES))

    fmt = ".2f" if normalized else "d"
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = format(val, fmt) if normalized else str(int(val))
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _detect_gradcam_layer(
    model: tf.keras.Model,
    preferred_name: str | None,
    model_key: str,
) -> str:
    """Return the name of the Grad-CAM target layer (for logging only)."""
    try:
        layer = find_conv_layer(model, preferred_name=preferred_name)
        return layer.name
    except Exception as exc:
        return f"detection_failed: {exc}"


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def _evaluate_one_model(
    config: dict,
    paths: list[str],
    y_true_full: np.ndarray,
    dataset_full: tf.data.Dataset,
    redundant_paths: set[str],
) -> dict:
    """Run full evaluation for one model and write all artefacts.

    Model inference is performed EXACTLY ONCE on the full 1600-image test set.
    The resulting prediction DataFrame is reused (filtered, not re-inferred)
    for the duplicate-sensitivity analysis.

    Returns a summary dict for inclusion in evaluation_summary.json.
    """
    key = config["key"]
    display_name = config["display_name"]
    model_path = config["path"]
    preferred_layer = config.get("gradcam_preferred_layer")

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {display_name}")
    print(f"  Model path: {model_path}")
    print(f"{'=' * 60}")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            f"Run the Phase 4/5 training scripts first."
        )

    model = tf.keras.models.load_model(str(model_path))
    print(f"  Model loaded. Layers: {len(model.layers)}")

    # --- Primary evaluation: ONE inference pass over all 1600 test images ---
    print(f"  Running inference on {len(paths)} test images …")
    y_true, y_pred, y_prob = _run_model_prediction(model, dataset_full)

    # Build predictions DataFrame — single source of truth for both primary
    # metrics and the sensitivity analysis (no re-inference).
    pred_df = _build_predictions_df(paths, y_true, y_pred, y_prob)

    metrics = compute_metrics(y_true, y_pred, class_names=CLASS_NAMES)
    report_df = compute_classification_report_df(
        y_true, y_pred, class_names=CLASS_NAMES
    )
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(CLASS_NAMES))
    cm_norm = normalize_confusion_matrix(cm)

    # Save artefacts
    out = RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics_path = out / f"{key}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump({**metrics, "model": display_name,
                   "dataset": "full official test set (1600 images)"}, fh, indent=2)
    print(f"  Metrics saved → {metrics_path.name}")

    # Classification report CSV
    report_path = out / f"{key}_classification_report.csv"
    report_df.to_csv(report_path)
    print(f"  Classification report saved → {report_path.name}")

    # Predictions CSV (written from DataFrame)
    pred_path = out / f"{key}_predictions.csv"
    _save_predictions_df(pred_df, pred_path)
    print(f"  Predictions saved → {pred_path.name}")

    # Confusion matrix CSV (raw counts)
    cm_df = pd.DataFrame(cm, index=list(CLASS_NAMES), columns=list(CLASS_NAMES))
    cm_csv_path = out / f"{key}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"  Confusion matrix CSV saved → {cm_csv_path.name}")

    # Confusion matrix PNGs
    _save_confusion_matrix_png(
        cm, f"{display_name}\nConfusion Matrix (counts)",
        out / f"{key}_confusion_matrix.png", normalized=False
    )
    _save_confusion_matrix_png(
        cm_norm, f"{display_name}\nConfusion Matrix (normalized)",
        out / f"{key}_confusion_matrix_normalized.png", normalized=True
    )
    print(f"  Confusion matrix PNGs saved.")

    # Print summary
    print(f"\n  PRIMARY RESULTS — {display_name}")
    print(f"    Accuracy:        {metrics['accuracy']:.4f}")
    print(f"    Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"    Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"    Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"    Weighted F1:     {metrics['weighted_f1']:.4f}")

    # --- Secondary: exact-duplicate sensitivity (filter pred_df, no re-inference) ---
    dedup_metrics = _compute_sensitivity_from_predictions(
        pred_df=pred_df,
        redundant_paths=redundant_paths,
        display_name=display_name,
        repo_root=REPO_ROOT,
    )
    # Persist per-model sensitivity JSON
    dup_json = DUP_RESULTS_DIR / f"{key}_metrics.json"
    DUP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with dup_json.open("w", encoding="utf-8") as fh:
        json.dump(dedup_metrics, fh, indent=2)

    # Detect Grad-CAM target layer name (structural inspection only)
    gradcam_layer_name = _detect_gradcam_layer(model, preferred_layer, key)
    print(f"  Grad-CAM target layer: {gradcam_layer_name}")

    return {
        "model_key": key,
        "display_name": display_name,
        "model_path": str(model_path),
        "preprocessing": config["preprocessing"],
        "primary_metrics": metrics,
        "dedup_metrics": dedup_metrics,
        "gradcam_target_layer": gradcam_layer_name,
        "gradcam_disclaimer": (
            "Grad-CAM highlights regions influencing the model's decision. "
            "It is NOT tumour segmentation and does NOT prove lesion location."
        ),
    }



def main() -> None:
    """Phase 6 final evaluation entry point."""

    _guard_flag()

    print()
    print("=" * 70)
    print("PHASE 6 — FINAL TEST EVALUATION")
    print("Brain Tumor MRI Classification (Educational/Research)")
    print("=" * 70)
    print()
    print("  IMPORTANT: This accesses the untouched official test set.")
    print("  The test set should be evaluated exactly ONCE.")
    print()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    eval_timestamp = datetime.now(tz=timezone.utc).isoformat()

    # -----------------------------------------------------------------------
    # GPU / device info
    # -----------------------------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    gpu_info = gpus[0].name if gpus else "CPU only"
    print(f"TensorFlow: {tf.__version__}")
    print(f"Device:     {gpu_info}")

    # -----------------------------------------------------------------------
    # Load test manifest
    # -----------------------------------------------------------------------
    if not TEST_MANIFEST.is_file():
        print(f"ERROR: test manifest not found: {TEST_MANIFEST}")
        print("Run 'python -m src.data.create_splits' first.")
        sys.exit(1)

    from src.data.data_pipeline import read_manifest_rows
    paths_full, labels_full = load_manifest(TEST_MANIFEST, root=REPO_ROOT)
    print(f"Test samples: {len(paths_full)}")

    dataset_full = build_dataset(
        TEST_MANIFEST, batch_size=BATCH_SIZE, training=False
    )

    # -----------------------------------------------------------------------
    # Load duplicate info for sensitivity analysis
    # -----------------------------------------------------------------------
    redundant_paths = _load_duplicate_paths(TEST_DUPS_CSV)
    if redundant_paths:
        print(
            f"Duplicate sensitivity: {len(redundant_paths)} redundant "
            "copies will be filtered from primary predictions (no re-inference)."
        )
    else:
        print(
            "Duplicate sensitivity: no redundant copies found "
            f"(checked {TEST_DUPS_CSV.name})."
        )

    # -----------------------------------------------------------------------
    # Evaluate each model
    # -----------------------------------------------------------------------
    summaries: list[dict] = []
    all_metrics: list[dict] = []
    all_names: list[str] = []

    for config in MODEL_CONFIGS:
        summary = _evaluate_one_model(
            config=config,
            paths=paths_full,
            y_true_full=labels_full,
            dataset_full=dataset_full,
            redundant_paths=redundant_paths,
        )
        summaries.append(summary)
        all_metrics.append(summary["primary_metrics"])
        all_names.append(summary["display_name"])

    # -----------------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------------
    name_a, name_b = all_names[0], all_names[1]
    comp_df = build_comparison_df(
        all_metrics[0], name_a, all_metrics[1], name_b
    )
    winner = select_winner(comp_df, primary_metric=PRIMARY_METRIC)

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(comp_df.to_string(float_format="{:.4f}".format))
    print(f"\nComparison rule: highest {PRIMARY_METRIC}")
    print(f"Selection: {winner['note']}")
    print(f"Selected model for Phase 7: {winner['selected_model']}")

    # Save comparison
    comp_csv = RESULTS_DIR / "model_comparison.csv"
    comp_df.to_csv(comp_csv)

    comp_json = RESULTS_DIR / "model_comparison.json"
    with comp_json.open("w", encoding="utf-8") as fh:
        comp_data = {
            "models": comp_df.reset_index().to_dict(orient="records"),
            "winner": winner,
            "primary_metric": PRIMARY_METRIC,
        }
        json.dump(comp_data, fh, indent=2)
    print(f"\n  Comparison saved → {comp_csv.name}, {comp_json.name}")

    # Sensitivity comparison CSV
    DUP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sens_rows = []
    for s in summaries:
        dm = s.get("dedup_metrics", {})
        sens_rows.append({
            "model": s["display_name"],
            "primary_accuracy": s["primary_metrics"].get("accuracy"),
            "primary_macro_f1": s["primary_metrics"].get("macro_f1"),
            "dedup_accuracy": dm.get("accuracy", None),
            "dedup_macro_f1": dm.get("macro_f1", None),
        })
    pd.DataFrame(sens_rows).to_csv(
        DUP_RESULTS_DIR / "comparison.csv", index=False
    )

    # -----------------------------------------------------------------------
    # evaluation_summary.json
    # -----------------------------------------------------------------------
    elapsed = time.monotonic() - start_time
    summary_doc = {
        "disclaimer": DISCLAIMER,
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "device": gpu_info,
        "eval_timestamp_utc": eval_timestamp,
        "official_test_evaluations_this_run": 1,
        "test_sample_count": len(paths_full),
        "class_mapping": LABEL_TO_ID,
        "class_names_fixed_order": list(CLASS_NAMES),
        "primary_comparison_metric": PRIMARY_METRIC,
        "models": [
            {
                "key": s["model_key"],
                "display_name": s["display_name"],
                "path": s["model_path"],
                "preprocessing": s["preprocessing"],
                "primary_metrics": s["primary_metrics"],
                "gradcam_target_layer": s["gradcam_target_layer"],
            }
            for s in summaries
        ],
        "model_selection": winner,
        "selected_model_for_phase7": winner["selected_model"],
        "gradcam_disclaimer": (
            "Grad-CAM highlights regions influencing the model's decision. "
            "It is NOT tumour segmentation and does NOT prove lesion location."
        ),
        "duplicate_sensitivity": {
            "test_duplicate_groups": 15,
            "redundant_copies_removed": len(redundant_paths),
            "note": (
                "Primary reported metrics use the full official test set. "
                "Sensitivity analysis is secondary only."
            ),
        },
        "total_evaluation_time_seconds": round(elapsed, 1),
    }

    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_doc, fh, indent=2)
    print(f"  Evaluation summary saved → {summary_path.name}")

    # -----------------------------------------------------------------------
    # Final banner
    # -----------------------------------------------------------------------
    mins, secs = divmod(int(elapsed), 60)
    print()
    print("=" * 70)
    print("Phase 6 evaluation complete.")
    print("=" * 70)
    for s in summaries:
        m = s["primary_metrics"]
        print(
            f"  {s['display_name']:30s} "
            f"acc={m['accuracy']:.4f}  "
            f"macro_f1={m['macro_f1']:.4f}"
        )
    print(f"  Selected for Phase 7: {winner['selected_model']}")
    print(f"  Runtime: {mins}m {secs}s")
    print(f"  Results: {RESULTS_DIR}/")
    print()
    print(f"  {DISCLAIMER}")
    print("=" * 70)


if __name__ == "__main__":
    main()
