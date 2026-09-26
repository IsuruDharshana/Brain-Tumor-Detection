"""
metrics.py
----------
Phase 6 metric computation utilities for the Brain Tumor MRI Classification
project.

Computes per-model and comparative metrics using scikit-learn.  All
functions operate on plain NumPy arrays (true labels and predicted labels /
probabilities) so they can be tested without loading any real data.

Public API
----------
    compute_metrics(y_true, y_pred, class_names) -> dict
    compute_classification_report_df(y_true, y_pred, class_names) -> DataFrame
    build_comparison_df(metrics_a, name_a, metrics_b, name_b) -> DataFrame
    select_winner(comparison_df, primary_metric) -> str
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Fixed class order for all outputs — never reordered.
_CLASS_ORDER = ("glioma", "meningioma", "notumor", "pituitary")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] = _CLASS_ORDER,
) -> dict:
    """Compute aggregate and per-class classification metrics.

    Args:
        y_true: integer ground-truth labels (0-based).
        y_pred: integer predicted labels (0-based).
        class_names: ordered class names matching the label integers.

    Returns:
        dict with keys:
            "accuracy"
            "macro_precision", "macro_recall", "macro_f1"
            "weighted_precision", "weighted_recall", "weighted_f1"
            "per_class": list of dicts, one per class, with keys
                "class", "precision", "recall", "f1", "support"
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    labels = list(range(len(class_names)))

    accuracy = float(accuracy_score(y_true, y_pred))

    macro_precision = float(
        precision_score(y_true, y_pred, labels=labels, average="macro",
                        zero_division=0)
    )
    macro_recall = float(
        recall_score(y_true, y_pred, labels=labels, average="macro",
                     zero_division=0)
    )
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="macro",
                 zero_division=0)
    )

    weighted_precision = float(
        precision_score(y_true, y_pred, labels=labels, average="weighted",
                        zero_division=0)
    )
    weighted_recall = float(
        recall_score(y_true, y_pred, labels=labels, average="weighted",
                     zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="weighted",
                 zero_division=0)
    )

    # Per-class metrics
    p_per = precision_score(y_true, y_pred, labels=labels, average=None,
                            zero_division=0)
    r_per = recall_score(y_true, y_pred, labels=labels, average=None,
                         zero_division=0)
    f_per = f1_score(y_true, y_pred, labels=labels, average=None,
                     zero_division=0)
    support_per = np.bincount(y_true, minlength=len(class_names))

    per_class = [
        {
            "class": class_names[i],
            "label_id": i,
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f_per[i]),
            "support": int(support_per[i]),
        }
        for i in range(len(class_names))
    ]

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def compute_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] = _CLASS_ORDER,
) -> pd.DataFrame:
    """Return a tidy DataFrame matching sklearn's classification report.

    Rows: one per class, plus macro avg and weighted avg.
    Columns: precision, recall, f1-score, support.

    Args:
        y_true: integer ground-truth labels.
        y_pred: integer predicted labels.
        class_names: ordered class names.

    Returns:
        pd.DataFrame with index = class name strings.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    labels = list(range(len(class_names)))

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    # Ensure integer support column where appropriate
    if "support" in df.columns:
        df["support"] = df["support"].astype(float)
    return df


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    """Return the raw integer confusion matrix (rows=true, cols=predicted).

    Fixed label order 0..num_classes-1.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion matrix (divide each row by its row sum).

    Rows with zero sum are left as zeros (avoids NaN).
    """
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero for empty rows.
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    return cm / safe_sums


def build_comparison_df(
    metrics_a: dict,
    name_a: str,
    metrics_b: dict,
    name_b: str,
) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame for two models.

    Args:
        metrics_a: output of compute_metrics() for model A.
        name_a: display name for model A (e.g. "Baseline CNN").
        metrics_b: output of compute_metrics() for model B.
        name_b: display name for model B.

    Returns:
        pd.DataFrame with one row per model and metric columns.
    """
    aggregate_keys = [
        "accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
    ]
    rows = []
    for name, metrics in [(name_a, metrics_a), (name_b, metrics_b)]:
        row = {"model": name}
        for k in aggregate_keys:
            row[k] = metrics.get(k, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    return df


def select_winner(
    comparison_df: pd.DataFrame,
    primary_metric: str = "macro_f1",
    secondary_metric: str = "accuracy",
) -> dict:
    """Determine the preferred model using a documented comparison rule.

    Rule:
        1. Highest primary_metric (default macro_f1) wins.
        2. If tied to 4 decimal places, fall back to secondary_metric.
        3. If still tied, the first model (index 0) is reported as tied.

    Args:
        comparison_df: output of build_comparison_df().
        primary_metric: column name of the primary comparison metric.
        secondary_metric: column name used only to break ties.

    Returns:
        dict with keys:
            "selected_model": str model name
            "primary_metric": str
            "primary_scores": dict model -> float
            "secondary_metric": str
            "secondary_scores": dict model -> float
            "is_tie": bool
            "note": str human-readable explanation
    """
    models = list(comparison_df.index)
    primary_scores = {m: float(comparison_df.loc[m, primary_metric]) for m in models}
    secondary_scores = {m: float(comparison_df.loc[m, secondary_metric]) for m in models}

    # Round to 4 d.p. for tie-detection (avoids floating-point noise)
    rounded = {m: round(primary_scores[m], 4) for m in models}
    best_primary = max(rounded.values())
    candidates = [m for m in models if rounded[m] == best_primary]

    is_tie = len(candidates) > 1
    if is_tie:
        # Break tie with secondary metric
        sec_rounded = {m: round(secondary_scores[m], 4) for m in candidates}
        best_sec = max(sec_rounded.values())
        final_candidates = [m for m in candidates if sec_rounded[m] == best_sec]
        selected = final_candidates[0]
        if len(final_candidates) > 1:
            note = (
                f"Models are tied on {primary_metric} "
                f"({best_primary:.4f}) and {secondary_metric} "
                f"({best_sec:.4f}). Reported first model as selected."
            )
        else:
            note = (
                f"Tied on {primary_metric} ({best_primary:.4f}). "
                f"Selected by {secondary_metric}: "
                f"{secondary_scores[selected]:.4f}."
            )
    else:
        selected = candidates[0]
        other = [m for m in models if m != selected]
        diff = best_primary - max(rounded[m] for m in other) if other else 0.0
        note = (
            f"Selected '{selected}' with {primary_metric}="
            f"{primary_scores[selected]:.4f} "
            f"(margin +{diff:.4f} over next best)."
        )

    return {
        "selected_model": selected,
        "primary_metric": primary_metric,
        "primary_scores": primary_scores,
        "secondary_metric": secondary_metric,
        "secondary_scores": secondary_scores,
        "is_tie": is_tie,
        "note": note,
    }
