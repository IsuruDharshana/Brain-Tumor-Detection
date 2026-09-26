"""
test_evaluation.py
------------------
Phase 6 unit tests for src/evaluation/metrics.py and
the duplicate-sensitivity helpers in src/evaluation/evaluate_models.py.

All tests use synthetic arrays or tiny fixture data.
NO real test-set images are loaded.
NO real model inference is run.
NO official 1600-image test set is accessed.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.evaluation.metrics import (
    build_comparison_df,
    compute_classification_report_df,
    compute_confusion_matrix,
    compute_metrics,
    normalize_confusion_matrix,
    select_winner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_preds() -> tuple[np.ndarray, np.ndarray]:
    """4-class, 8 samples; model predicts perfectly."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    return y_true, y_pred


@pytest.fixture
def imperfect_preds() -> tuple[np.ndarray, np.ndarray]:
    """4-class, 8 samples; model makes two errors."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    y_pred = np.array([0, 1, 1, 1, 2, 0, 3, 3], dtype=np.int32)
    return y_true, y_pred


@pytest.fixture
def toy_probs(imperfect_preds) -> np.ndarray:
    """Simple probability array consistent with imperfect_preds."""
    y_true, y_pred = imperfect_preds
    n = len(y_pred)
    probs = np.zeros((n, 4), dtype=np.float32)
    for i, pred in enumerate(y_pred):
        probs[i, pred] = 0.9
        remainder = 0.1 / 3
        for j in range(4):
            if j != pred:
                probs[i, j] = remainder
    return probs


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_perfect_accuracy(perfect_preds):
    y_true, y_pred = perfect_preds
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(1.0)


def test_perfect_macro_f1(perfect_preds):
    y_true, y_pred = perfect_preds
    m = compute_metrics(y_true, y_pred)
    assert m["macro_f1"] == pytest.approx(1.0)


def test_imperfect_accuracy_below_one(imperfect_preds):
    y_true, y_pred = imperfect_preds
    m = compute_metrics(y_true, y_pred)
    assert 0.0 < m["accuracy"] < 1.0


def test_imperfect_macro_f1_below_one(imperfect_preds):
    y_true, y_pred = imperfect_preds
    m = compute_metrics(y_true, y_pred)
    assert 0.0 < m["macro_f1"] < 1.0


def test_metrics_keys_present(perfect_preds):
    y_true, y_pred = perfect_preds
    m = compute_metrics(y_true, y_pred)
    for key in [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1", "per_class"
    ]:
        assert key in m, f"Missing key: {key}"


def test_per_class_count(perfect_preds):
    y_true, y_pred = perfect_preds
    m = compute_metrics(y_true, y_pred)
    assert len(m["per_class"]) == 4


def test_per_class_fixed_order(perfect_preds):
    y_true, y_pred = perfect_preds
    m = compute_metrics(y_true, y_pred)
    expected = ("glioma", "meningioma", "notumor", "pituitary")
    names = tuple(c["class"] for c in m["per_class"])
    assert names == expected, f"Class order wrong: {names}"


def test_per_class_has_required_fields(perfect_preds):
    y_true, y_pred = perfect_preds
    m = compute_metrics(y_true, y_pred)
    for cls in m["per_class"]:
        for field in ("class", "label_id", "precision", "recall", "f1", "support"):
            assert field in cls


def test_zero_division_handled():
    """A class with no predictions should not raise; zero_division=0."""
    # All predicted as class 0
    y_true = np.array([0, 1, 2, 3], dtype=np.int32)
    y_pred = np.array([0, 0, 0, 0], dtype=np.int32)
    m = compute_metrics(y_true, y_pred)
    # macro_precision still computes without error
    assert "macro_precision" in m
    assert np.isfinite(m["macro_precision"])


def test_all_same_class_no_crash():
    """Edge case: all true labels are class 0."""
    y_true = np.array([0] * 8, dtype=np.int32)
    y_pred = np.array([0] * 8, dtype=np.int32)
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(1.0)


def test_metrics_values_in_range(imperfect_preds):
    """All aggregate metrics must be in [0, 1]."""
    y_true, y_pred = imperfect_preds
    m = compute_metrics(y_true, y_pred)
    for key in ["accuracy", "macro_f1", "macro_precision", "macro_recall",
                "weighted_f1", "weighted_precision", "weighted_recall"]:
        assert 0.0 <= m[key] <= 1.0, f"{key} = {m[key]} out of [0,1]"


# ---------------------------------------------------------------------------
# Classification report DataFrame
# ---------------------------------------------------------------------------

def test_classification_report_is_dataframe(perfect_preds):
    y_true, y_pred = perfect_preds
    df = compute_classification_report_df(y_true, y_pred)
    assert isinstance(df, pd.DataFrame)


def test_classification_report_has_class_rows(perfect_preds):
    y_true, y_pred = perfect_preds
    df = compute_classification_report_df(y_true, y_pred)
    for cls in ("glioma", "meningioma", "notumor", "pituitary"):
        assert cls in df.index, f"Class '{cls}' missing from report index"


def test_classification_report_fixed_order(perfect_preds):
    """First four rows must appear in fixed class order."""
    y_true, y_pred = perfect_preds
    df = compute_classification_report_df(y_true, y_pred)
    rows = list(df.index)
    expected = ["glioma", "meningioma", "notumor", "pituitary"]
    assert rows[:4] == expected, f"Wrong order: {rows[:4]}"


def test_classification_report_has_precision_column(perfect_preds):
    y_true, y_pred = perfect_preds
    df = compute_classification_report_df(y_true, y_pred)
    assert "precision" in df.columns


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def test_confusion_matrix_shape(perfect_preds):
    y_true, y_pred = perfect_preds
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=4)
    assert cm.shape == (4, 4)


def test_confusion_matrix_diagonal_perfect(perfect_preds):
    y_true, y_pred = perfect_preds
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=4)
    assert np.all(cm == np.diag(np.diag(cm))), "Off-diagonal should be zero for perfect preds"


def test_confusion_matrix_sum_equals_n(imperfect_preds):
    y_true, y_pred = imperfect_preds
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=4)
    assert cm.sum() == len(y_true)


def test_normalized_confusion_matrix_rows_sum_to_one(imperfect_preds):
    y_true, y_pred = imperfect_preds
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=4)
    cm_norm = normalize_confusion_matrix(cm)
    row_sums = cm_norm.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-6)


def test_normalized_confusion_matrix_range(imperfect_preds):
    y_true, y_pred = imperfect_preds
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=4)
    cm_norm = normalize_confusion_matrix(cm)
    assert np.all(cm_norm >= 0.0)
    assert np.all(cm_norm <= 1.0 + 1e-8)


def test_normalize_zero_row_safe():
    """A confusion matrix with a zero row must not produce NaN."""
    cm = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])
    cm_norm = normalize_confusion_matrix(cm)
    assert np.all(np.isfinite(cm_norm)), "NaN in normalized CM with zero row"
    assert np.all(cm_norm[1] == 0.0), "Zero row should remain zero"


# ---------------------------------------------------------------------------
# Comparison table and winner selection
# ---------------------------------------------------------------------------

def _make_metrics(accuracy, macro_f1):
    """Return a minimal metrics dict for comparison tests."""
    return {
        "accuracy": accuracy,
        "macro_precision": macro_f1,
        "macro_recall": macro_f1,
        "macro_f1": macro_f1,
        "weighted_precision": macro_f1,
        "weighted_recall": macro_f1,
        "weighted_f1": macro_f1,
        "per_class": [],
    }


def test_comparison_df_has_both_models():
    m1 = _make_metrics(0.90, 0.88)
    m2 = _make_metrics(0.85, 0.82)
    df = build_comparison_df(m1, "ModelA", m2, "ModelB")
    assert "ModelA" in df.index
    assert "ModelB" in df.index


def test_comparison_df_columns():
    m1 = _make_metrics(0.90, 0.88)
    m2 = _make_metrics(0.85, 0.82)
    df = build_comparison_df(m1, "ModelA", m2, "ModelB")
    assert "macro_f1" in df.columns
    assert "accuracy" in df.columns


def test_winner_selects_higher_macro_f1():
    m1 = _make_metrics(0.90, 0.88)
    m2 = _make_metrics(0.85, 0.82)
    df = build_comparison_df(m1, "ModelA", m2, "ModelB")
    result = select_winner(df, primary_metric="macro_f1")
    assert result["selected_model"] == "ModelA"


def test_winner_tie_uses_accuracy():
    m1 = _make_metrics(0.90, 0.88)
    m2 = _make_metrics(0.85, 0.88)  # same macro_f1
    df = build_comparison_df(m1, "ModelA", m2, "ModelB")
    result = select_winner(df, primary_metric="macro_f1", secondary_metric="accuracy")
    assert result["selected_model"] == "ModelA"
    assert result["is_tie"] is True


def test_winner_result_has_required_keys():
    m1 = _make_metrics(0.90, 0.88)
    m2 = _make_metrics(0.85, 0.82)
    df = build_comparison_df(m1, "ModelA", m2, "ModelB")
    result = select_winner(df)
    for key in ("selected_model", "primary_metric", "primary_scores",
                "is_tie", "note"):
        assert key in result


# ---------------------------------------------------------------------------
# Duplicate-removal logic (testing _load_duplicate_paths and _build_dedup_manifest)
# ---------------------------------------------------------------------------

def test_load_duplicate_paths_returns_set(tmp_path):
    """_load_duplicate_paths must return only is_representative=False paths."""
    from src.evaluation.evaluate_models import _load_duplicate_paths
    dup_csv = tmp_path / "test_duplicates.csv"
    dup_csv.write_text(
        "group_id,sha256,relative_path,label_name,is_representative,group_size\n"
        "1,abc,data/raw/Testing/glioma/img1.jpg,glioma,True,2\n"
        "1,abc,data/raw/Testing/glioma/img2.jpg,glioma,False,2\n"
        "2,def,data/raw/Testing/meningioma/img3.jpg,meningioma,True,2\n"
        "2,def,data/raw/Testing/meningioma/img4.jpg,meningioma,False,2\n",
        encoding="utf-8",
    )
    redundant = _load_duplicate_paths(dup_csv)
    assert isinstance(redundant, set)
    assert "data/raw/Testing/glioma/img2.jpg" in redundant
    assert "data/raw/Testing/meningioma/img4.jpg" in redundant
    # Representatives must NOT be in the redundant set
    assert "data/raw/Testing/glioma/img1.jpg" not in redundant


def test_load_duplicate_paths_empty_file(tmp_path):
    """_load_duplicate_paths on an empty file returns an empty set."""
    from src.evaluation.evaluate_models import _load_duplicate_paths
    dup_csv = tmp_path / "empty_dups.csv"
    dup_csv.write_text("group_id,sha256,relative_path,label_name,is_representative,group_size\n")
    redundant = _load_duplicate_paths(dup_csv)
    assert redundant == set()


def test_load_duplicate_paths_missing_file(tmp_path):
    """_load_duplicate_paths on a non-existent file returns an empty set."""
    from src.evaluation.evaluate_models import _load_duplicate_paths
    result = _load_duplicate_paths(tmp_path / "does_not_exist.csv")
    assert result == set()


def test_filter_predictions_df_removes_redundant_rows(tmp_path):
    """_filter_predictions_df must exclude only redundant-path rows."""
    from src.evaluation.evaluate_models import _filter_predictions_df
    repo = tmp_path / "repo"
    repo.mkdir()
    pred_df = pd.DataFrame({
        "filepath": [
            str(repo / "data/raw/Testing/glioma/img1.jpg"),
            str(repo / "data/raw/Testing/glioma/img2.jpg"),   # redundant
            str(repo / "data/raw/Testing/meningioma/img3.jpg"),
        ],
        "true_label_id": [0, 0, 1],
        "predicted_label_id": [0, 0, 1],
        "confidence": [0.9, 0.9, 0.9],
    })
    redundant = {"data/raw/Testing/glioma/img2.jpg"}
    filtered = _filter_predictions_df(pred_df, redundant, repo_root=repo)
    assert len(filtered) == 2
    paths = set(filtered["filepath"].tolist())
    assert str(repo / "data/raw/Testing/glioma/img1.jpg") in paths
    assert str(repo / "data/raw/Testing/glioma/img2.jpg") not in paths
    assert str(repo / "data/raw/Testing/meningioma/img3.jpg") in paths


def test_filter_predictions_df_no_redundant_keeps_all(tmp_path):
    """With an empty redundant set, all rows are preserved."""
    from src.evaluation.evaluate_models import _filter_predictions_df
    repo = tmp_path / "repo"
    pred_df = pd.DataFrame({
        "filepath": [str(repo / "a.jpg"), str(repo / "b.jpg")],
        "true_label_id": [0, 1],
        "predicted_label_id": [0, 1],
        "confidence": [0.9, 0.8],
    })
    filtered = _filter_predictions_df(pred_df, set(), repo_root=repo)
    assert len(filtered) == 2


def test_filter_predictions_df_does_not_mutate_original(tmp_path):
    """_filter_predictions_df must NOT mutate the original DataFrame."""
    from src.evaluation.evaluate_models import _filter_predictions_df
    repo = tmp_path / "repo"
    repo.mkdir()
    pred_df = pd.DataFrame({
        "filepath": [
            str(repo / "data/raw/Testing/glioma/img1.jpg"),
            str(repo / "data/raw/Testing/glioma/img2.jpg"),
        ],
        "true_label_id": [0, 0],
        "predicted_label_id": [0, 0],
        "confidence": [0.9, 0.9],
    })
    original_len = len(pred_df)
    original_paths = list(pred_df["filepath"])
    redundant = {"data/raw/Testing/glioma/img2.jpg"}
    _filter_predictions_df(pred_df, redundant, repo_root=repo)
    # Original must be unchanged
    assert len(pred_df) == original_len
    assert list(pred_df["filepath"]) == original_paths


def test_compute_sensitivity_no_model_parameter():
    """_compute_sensitivity_from_predictions must NOT accept a 'model' argument."""
    import inspect
    from src.evaluation.evaluate_models import _compute_sensitivity_from_predictions
    sig = inspect.signature(_compute_sensitivity_from_predictions)
    params = list(sig.parameters.keys())
    assert "model" not in params, (
        "Sensitivity function must not require a model — "
        "it should operate on the prediction DataFrame only."
    )
    assert "dataset" not in params, (
        "Sensitivity function must not require a dataset."
    )


def test_compute_sensitivity_no_redundant_returns_note():
    """With empty redundant_paths, sensitivity returns a note and skips analysis."""
    from pathlib import Path
    from src.evaluation.evaluate_models import _compute_sensitivity_from_predictions
    pred_df = pd.DataFrame({
        "filepath": ["/repo/a.jpg", "/repo/b.jpg"],
        "true_label_id": [0, 1],
        "predicted_label_id": [0, 1],
        "confidence": [0.9, 0.9],
    })
    result = _compute_sensitivity_from_predictions(
        pred_df=pred_df,
        redundant_paths=set(),
        display_name="TestModel",
        repo_root=Path("/repo"),
    )
    assert "note" in result
    assert "skipped" in result["note"].lower()


def test_compute_sensitivity_filters_and_recomputes(tmp_path):
    """Sensitivity must compute metrics from filtered rows only."""
    from src.evaluation.evaluate_models import _compute_sensitivity_from_predictions
    repo = tmp_path / "repo"
    repo.mkdir()
    pred_df = pd.DataFrame({
        "filepath": [
            str(repo / "data/raw/Testing/glioma/img1.jpg"),
            str(repo / "data/raw/Testing/glioma/img2.jpg"),  # redundant
            str(repo / "data/raw/Testing/notumor/img3.jpg"),
            str(repo / "data/raw/Testing/pituitary/img4.jpg"),
        ],
        "true_label_id":      [0, 0, 2, 3],
        "predicted_label_id": [0, 1, 2, 3],  # img2 mis-predicted as 1
        "confidence": [0.9, 0.8, 0.9, 0.9],
    })
    redundant = {"data/raw/Testing/glioma/img2.jpg"}
    result = _compute_sensitivity_from_predictions(
        pred_df=pred_df,
        redundant_paths=redundant,
        display_name="TestModel",
        repo_root=repo,
    )
    # After removing img2 (the mis-predicted one), accuracy should be 1.0
    assert result.get("accuracy") == pytest.approx(1.0), (
        f"Expected accuracy 1.0 after removing the one error, got {result.get('accuracy')}"
    )
    assert "SECONDARY" in result.get("note", "")
    assert result.get("model") == "TestModel"


def test_compute_sensitivity_does_not_mutate_pred_df(tmp_path):
    """Sensitivity analysis must not mutate the input pred_df."""
    from src.evaluation.evaluate_models import _compute_sensitivity_from_predictions
    repo = tmp_path / "repo"
    repo.mkdir()
    pred_df = pd.DataFrame({
        "filepath": [
            str(repo / "data/raw/Testing/glioma/img1.jpg"),
            str(repo / "data/raw/Testing/glioma/img2.jpg"),
        ],
        "true_label_id": [0, 0],
        "predicted_label_id": [0, 0],
        "confidence": [0.9, 0.9],
    })
    original_len = len(pred_df)
    redundant = {"data/raw/Testing/glioma/img2.jpg"}
    _compute_sensitivity_from_predictions(
        pred_df=pred_df,
        redundant_paths=redundant,
        display_name="TestModel",
        repo_root=repo,
    )
    assert len(pred_df) == original_len  # original unchanged


def test_prediction_table_schema(toy_probs, imperfect_preds):
    """_build_predictions_df and _save_predictions_df must produce expected columns."""
    from src.evaluation.evaluate_models import _build_predictions_df, _save_predictions_df
    y_true, y_pred = imperfect_preds
    paths = [f"img_{i}.jpg" for i in range(len(y_true))]
    pred_df = _build_predictions_df(paths, y_true, y_pred, toy_probs)
    out = Path(__import__("tempfile").mkdtemp()) / "preds.csv"
    _save_predictions_df(pred_df, out)
    assert out.is_file()
    with out.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    for col in ("filepath", "true_label_id", "true_label_name",
                "predicted_label_id", "predicted_label_name", "confidence",
                "prob_glioma", "prob_meningioma", "prob_notumor", "prob_pituitary"):
        assert col in fieldnames, f"Missing column: {col}"
    assert len(rows) == len(y_true)


def test_confidence_is_max_prob(toy_probs, imperfect_preds):
    """Confidence column must equal the predicted class probability."""
    from src.evaluation.evaluate_models import _build_predictions_df
    y_true, y_pred = imperfect_preds
    paths = [f"img_{i}.jpg" for i in range(len(y_true))]
    pred_df = _build_predictions_df(paths, y_true, y_pred, toy_probs)
    for i, row in pred_df.iterrows():
        pred_id = int(row["predicted_label_id"])
        expected_conf = float(toy_probs[i, pred_id])
        assert abs(float(row["confidence"]) - expected_conf) < 1e-5


def test_compute_sensitivity_preserves_class_order(tmp_path):
    """Sensitivity metrics must preserve the fixed 4-class ordering."""
    from src.evaluation.evaluate_models import _compute_sensitivity_from_predictions
    repo = tmp_path / "repo"
    repo.mkdir()
    pred_df = pd.DataFrame({
        "filepath": [
            str(repo / "data/raw/Testing/glioma/img1.jpg"),
            str(repo / "data/raw/Testing/glioma/img2.jpg"),
            str(repo / "data/raw/Testing/notumor/img3.jpg"),
        ],
        "true_label_id": [0, 0, 2],
        "predicted_label_id": [0, 0, 2],
        "confidence": [0.9, 0.9, 0.9],
    })
    redundant = {"data/raw/Testing/glioma/img2.jpg"}
    result = _compute_sensitivity_from_predictions(
        pred_df=pred_df,
        redundant_paths=redundant,
        display_name="TestModel",
        repo_root=repo,
    )
    expected_order = ["glioma", "meningioma", "notumor", "pituitary"]
    observed_order = [c["class"] for c in result["per_class"]]
    assert observed_order == expected_order
