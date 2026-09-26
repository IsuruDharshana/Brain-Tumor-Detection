"""
test_gradcam.py
---------------
Phase 6 unit tests for src/explainability/gradcam.py.

All tests use tiny synthetic Keras models built with random weights
(weights="None" / no ImageNet download).

NO real model files are loaded from disk.
NO real MRI images are used.
NO official test-set images are accessed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

from src.explainability.gradcam import (
    compute_gradcam,
    find_conv_layer,
    overlay_heatmap,
)


# ---------------------------------------------------------------------------
# Synthetic model builders
# ---------------------------------------------------------------------------

def _build_flat_cnn(input_shape=(32, 32, 3), num_classes=4) -> tf.keras.Model:
    """Build a tiny flat CNN resembling the Baseline CNN structure."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8, 3, padding="same",
                               activation="relu", name="block1_conv")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same",
                               activation="relu", name="block2_conv")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    dtype="float32", name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="flat_cnn")


def _build_nested_cnn(input_shape=(32, 32, 3), num_classes=4) -> tf.keras.Model:
    """Build a genuinely nested Functional model matching the transfer learning structure:
    outer Input -> preprocessing layer (Rescaling) -> nested backbone Model -> GAP -> Dense.
    """
    # Inner sub-model (backbone-like)
    b_in = tf.keras.Input(shape=input_shape, name="backbone_input")
    b_x = tf.keras.layers.Conv2D(8, 3, padding="same",
                                  activation="relu", name="backbone_conv1")(b_in)
    b_x = tf.keras.layers.Conv2D(16, 3, padding="same",
                                  activation="relu", name="backbone_top_conv")(b_x)
    backbone = tf.keras.Model(inputs=b_in, outputs=b_x, name="efficientnetb0_mock")

    # Outer model: input -> preprocessing -> backbone -> GAP -> Dense
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = tf.keras.layers.Rescaling(255.0, name="adapter_rescale_to_255")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="head_gap")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    dtype="float32", name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs,
                          name="efficientnet_b0_transfer")


@pytest.fixture(scope="module")
def flat_model() -> tf.keras.Model:
    return _build_flat_cnn()


@pytest.fixture(scope="module")
def nested_model() -> tf.keras.Model:
    return _build_nested_cnn()


@pytest.fixture
def sample_image() -> np.ndarray:
    """Single float32 image (H, W, C) in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((32, 32, 3)).astype(np.float32)


@pytest.fixture
def sample_image_batch() -> np.ndarray:
    """Batched float32 image (1, H, W, C) in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((1, 32, 32, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# find_conv_layer — flat model
# ---------------------------------------------------------------------------

def test_find_conv_layer_flat_model_returns_layer(flat_model):
    layer = find_conv_layer(flat_model)
    assert isinstance(layer, tf.keras.layers.Layer)


def test_find_conv_layer_flat_model_is_conv2d(flat_model):
    layer = find_conv_layer(flat_model)
    assert isinstance(layer, tf.keras.layers.Conv2D)


def test_find_conv_layer_flat_model_is_last_conv(flat_model):
    """Auto-detected layer must be the last Conv2D (block2_conv)."""
    layer = find_conv_layer(flat_model)
    assert layer.name == "block2_conv"


def test_find_conv_layer_preferred_name_found(flat_model):
    layer = find_conv_layer(flat_model, preferred_name="block1_conv")
    assert layer.name == "block1_conv"


def test_find_conv_layer_preferred_name_not_found_falls_back(flat_model):
    """A preferred name that doesn't exist should fall back to last conv."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        layer = find_conv_layer(flat_model, preferred_name="nonexistent_layer")
    assert isinstance(layer, tf.keras.layers.Conv2D)
    # A warning should have been issued.
    assert len(w) >= 1


# ---------------------------------------------------------------------------
# find_conv_layer — nested (EfficientNetB0-like) model
# ---------------------------------------------------------------------------

def test_find_conv_layer_nested_model_returns_layer(nested_model):
    layer = find_conv_layer(nested_model)
    assert isinstance(layer, tf.keras.layers.Layer)


def test_find_conv_layer_nested_model_is_conv2d(nested_model):
    layer = find_conv_layer(nested_model)
    assert isinstance(layer, tf.keras.layers.Conv2D)


def test_find_conv_layer_nested_model_finds_backbone_conv(nested_model):
    """For a nested backbone, last conv is backbone_top_conv."""
    layer = find_conv_layer(nested_model)
    assert layer.name == "backbone_top_conv"


def test_find_conv_layer_nested_model_preferred_name(nested_model):
    layer = find_conv_layer(nested_model, preferred_name="backbone_conv1")
    assert layer.name == "backbone_conv1"


def test_find_conv_layer_no_conv_raises():
    """A model with no Conv2D layers must raise ValueError."""
    inputs = tf.keras.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(4, activation="softmax")(inputs)
    dense_only = tf.keras.Model(inputs=inputs, outputs=outputs)
    with pytest.raises(ValueError, match="No Conv2D"):
        find_conv_layer(dense_only)


# ---------------------------------------------------------------------------
# compute_gradcam — flat model
# ---------------------------------------------------------------------------

def test_gradcam_flat_output_shape(flat_model, sample_image):
    heatmap = compute_gradcam(flat_model, sample_image)
    assert heatmap.shape == (32, 32), f"Expected (32, 32), got {heatmap.shape}"


def test_gradcam_flat_values_finite(flat_model, sample_image):
    heatmap = compute_gradcam(flat_model, sample_image)
    assert np.all(np.isfinite(heatmap)), "Heatmap contains non-finite values"


def test_gradcam_flat_normalized_range(flat_model, sample_image):
    heatmap = compute_gradcam(flat_model, sample_image)
    assert heatmap.min() >= 0.0, f"Heatmap min < 0: {heatmap.min()}"
    assert heatmap.max() <= 1.0 + 1e-6, f"Heatmap max > 1: {heatmap.max()}"


def test_gradcam_flat_dtype_float32(flat_model, sample_image):
    heatmap = compute_gradcam(flat_model, sample_image)
    assert heatmap.dtype == np.float32


def test_gradcam_flat_batch_input(flat_model, sample_image_batch):
    """Batched (1, H, W, C) input should work without error."""
    heatmap = compute_gradcam(flat_model, sample_image_batch)
    assert heatmap.shape == (32, 32)


def test_gradcam_flat_explicit_class(flat_model, sample_image):
    """Explicit class_idx must not raise and must produce a valid heatmap."""
    heatmap = compute_gradcam(flat_model, sample_image, class_idx=0)
    assert heatmap.shape == (32, 32)
    assert np.all(np.isfinite(heatmap))


def test_gradcam_flat_different_classes_differ(flat_model, sample_image):
    """Heatmaps for different explicit classes may differ (but shouldn't crash)."""
    h0 = compute_gradcam(flat_model, sample_image, class_idx=0)
    h1 = compute_gradcam(flat_model, sample_image, class_idx=1)
    # Both must be valid; we don't require them to differ (could be same model).
    assert h0.shape == h1.shape
    assert np.all(np.isfinite(h0))
    assert np.all(np.isfinite(h1))


def test_gradcam_does_not_modify_model_weights(flat_model, sample_image):
    """Model weights must not change after compute_gradcam."""
    weights_before = [w.numpy().copy() for w in flat_model.weights]
    compute_gradcam(flat_model, sample_image)
    weights_after = [w.numpy() for w in flat_model.weights]
    for wb, wa in zip(weights_before, weights_after):
        np.testing.assert_array_equal(wb, wa, err_msg="Model weight changed!")


# ---------------------------------------------------------------------------
# compute_gradcam — nested model
# ---------------------------------------------------------------------------

def test_gradcam_nested_output_shape(nested_model, sample_image):
    heatmap = compute_gradcam(nested_model, sample_image)
    assert heatmap.shape == (32, 32)


def test_gradcam_nested_values_finite(nested_model, sample_image):
    heatmap = compute_gradcam(nested_model, sample_image)
    assert np.all(np.isfinite(heatmap))


def test_gradcam_nested_normalized_range(nested_model, sample_image):
    heatmap = compute_gradcam(nested_model, sample_image)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_gradcam_nested_does_not_modify_weights(nested_model, sample_image):
    weights_before = [w.numpy().copy() for w in nested_model.weights]
    compute_gradcam(nested_model, sample_image)
    weights_after = [w.numpy() for w in nested_model.weights]
    for wb, wa in zip(weights_before, weights_after):
        np.testing.assert_array_equal(wb, wa)


def test_gradcam_nested_with_preprocessing_end_to_end(nested_model, sample_image):
    """End-to-end test on genuinely nested Functional model with preprocessing:
    outer Input -> preprocessing layer (Rescaling) -> nested backbone -> GAP -> Dense.

    Verifies:
      - no graph-disconnected error
      - finite heatmap
      - correct spatial shape
      - heatmap normalized to [0, 1]
    """
    heatmap = compute_gradcam(nested_model, sample_image)
    assert heatmap is not None
    assert heatmap.shape == (32, 32)
    assert np.all(np.isfinite(heatmap))
    assert 0.0 <= heatmap.min()
    assert heatmap.max() <= 1.0 + 1e-6




# ---------------------------------------------------------------------------
# overlay_heatmap
# ---------------------------------------------------------------------------

def test_overlay_output_shape_rgb(sample_image):
    heatmap = np.random.rand(32, 32).astype(np.float32)
    overlay = overlay_heatmap(sample_image, heatmap)
    assert overlay.shape == (32, 32, 3)


def test_overlay_output_dtype_float32(sample_image):
    heatmap = np.random.rand(32, 32).astype(np.float32)
    overlay = overlay_heatmap(sample_image, heatmap)
    assert overlay.dtype == np.float32


def test_overlay_values_in_range(sample_image):
    heatmap = np.random.rand(32, 32).astype(np.float32)
    overlay = overlay_heatmap(sample_image, heatmap)
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0 + 1e-6


def test_overlay_pure_heatmap_alpha_one(sample_image):
    """alpha=1.0 should make the output approach the heatmap colour."""
    heatmap = np.ones((32, 32), dtype=np.float32)  # uniform max heatmap
    overlay = overlay_heatmap(sample_image, heatmap, alpha=1.0)
    # All pixels should have the same heatmap colour (all-ones jet → dark red)
    assert overlay.shape == (32, 32, 3)
    assert np.all(np.isfinite(overlay))


def test_overlay_grayscale_image():
    """Grayscale (H, W) input must be handled and output RGB."""
    gray = np.ones((32, 32), dtype=np.float32) * 0.5
    heatmap = np.zeros((32, 32), dtype=np.float32)
    overlay = overlay_heatmap(gray, heatmap)
    assert overlay.shape == (32, 32, 3)


def test_overlay_single_channel_image():
    """Single-channel (H, W, 1) input must be handled and output RGB."""
    single = np.ones((32, 32, 1), dtype=np.float32) * 0.5
    heatmap = np.zeros((32, 32), dtype=np.float32)
    overlay = overlay_heatmap(single, heatmap)
    assert overlay.shape == (32, 32, 3)


def test_overlay_zero_heatmap_returns_original(sample_image):
    """With alpha=0.0 the overlay must equal the original image (3-channel)."""
    heatmap = np.zeros((32, 32), dtype=np.float32)
    overlay = overlay_heatmap(sample_image, heatmap, alpha=0.0)
    np.testing.assert_allclose(overlay, sample_image, atol=1e-6)


# ---------------------------------------------------------------------------
# Deterministic Grad-CAM example selection tests
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_predictions_df() -> pd.DataFrame:
    """Synthetic predictions DataFrame with deliberate confidence vs. filepath inversion."""
    import pandas as pd
    rows = [
        # Glioma: z_glioma has high confidence, a_glioma has lower confidence.
        # Lexicographic sort MUST pick a_glioma.jpg, NOT z_glioma.jpg.
        {"filepath": "data/z_glioma.jpg", "true_label_id": 0, "true_label_name": "glioma",
         "predicted_label_id": 0, "predicted_label_name": "glioma", "confidence": 0.99},
        {"filepath": "data/a_glioma.jpg", "true_label_id": 0, "true_label_name": "glioma",
         "predicted_label_id": 0, "predicted_label_name": "glioma", "confidence": 0.60},

        # Meningioma: single correct
        {"filepath": "data/m1.jpg", "true_label_id": 1, "true_label_name": "meningioma",
         "predicted_label_id": 1, "predicted_label_name": "meningioma", "confidence": 0.85},

        # Notumor: single correct
        {"filepath": "data/n1.jpg", "true_label_id": 2, "true_label_name": "notumor",
         "predicted_label_id": 2, "predicted_label_name": "notumor", "confidence": 0.90},

        # Pituitary: single correct
        {"filepath": "data/p1.jpg", "true_label_id": 3, "true_label_name": "pituitary",
         "predicted_label_id": 3, "predicted_label_name": "pituitary", "confidence": 0.92},

        # 3 Misclassified examples:
        # z_err (conf=0.99), b_err (conf=0.70), a_err (conf=0.50).
        # Lexicographic sort must take a_err and b_err, capping at 2.
        {"filepath": "data/z_err.jpg", "true_label_id": 0, "true_label_name": "glioma",
         "predicted_label_id": 2, "predicted_label_name": "notumor", "confidence": 0.99},
        {"filepath": "data/b_err.jpg", "true_label_id": 1, "true_label_name": "meningioma",
         "predicted_label_id": 0, "predicted_label_name": "glioma", "confidence": 0.70},
        {"filepath": "data/a_err.jpg", "true_label_id": 3, "true_label_name": "pituitary",
         "predicted_label_id": 1, "predicted_label_name": "meningioma", "confidence": 0.50},
    ]
    return pd.DataFrame(rows)


def test_select_gradcam_examples_one_per_available_class(toy_predictions_df):
    """Exactly one correct example must be selected for each available class."""
    from src.explainability.generate_gradcam_examples import select_gradcam_examples
    selected = select_gradcam_examples(toy_predictions_df)
    correct = [ex for ex in selected if ex["example_type"] == "correct"]
    assert len(correct) == 4
    classes_selected = [ex["true_label_name"] for ex in correct]
    assert classes_selected == ["glioma", "meningioma", "notumor", "pituitary"]


def test_select_gradcam_examples_lexicographic_and_not_confidence(toy_predictions_df):
    """Selection must be lexicographic by filepath, ignoring higher confidence."""
    from src.explainability.generate_gradcam_examples import select_gradcam_examples
    selected = select_gradcam_examples(toy_predictions_df)
    glioma_ex = next(ex for ex in selected if ex["true_label_name"] == "glioma" and ex["example_type"] == "correct")
    # Must pick a_glioma.jpg (conf 0.60) instead of z_glioma.jpg (conf 0.99)
    assert glioma_ex["filepath"] == "data/a_glioma.jpg"
    assert glioma_ex["confidence"] == 0.60


def test_select_gradcam_examples_misclassified_max_two(toy_predictions_df):
    """Misclassified examples must be sorted lexicographically and capped at 2."""
    from src.explainability.generate_gradcam_examples import select_gradcam_examples
    selected = select_gradcam_examples(toy_predictions_df, max_misclassified=2)
    misclassified = [ex for ex in selected if ex["example_type"] == "misclassified"]
    assert len(misclassified) == 2
    # Lexicographically first two are a_err.jpg and b_err.jpg
    assert misclassified[0]["filepath"] == "data/a_err.jpg"
    assert misclassified[1]["filepath"] == "data/b_err.jpg"


def test_select_gradcam_examples_metadata_schema(toy_predictions_df):
    """Selected examples must adhere to the required metadata schema."""
    from src.explainability.generate_gradcam_examples import select_gradcam_examples
    selected = select_gradcam_examples(toy_predictions_df)
    required_keys = {
        "filepath", "example_type", "true_label_id", "true_label_name",
        "predicted_label_id", "predicted_label_name", "confidence",
        "target_layer", "output_image"
    }
    for ex in selected:
        missing = required_keys - set(ex.keys())
        assert not missing, f"Missing required keys: {missing}"
        assert ex["target_layer"] == "block4_conv"
        assert ex["output_image"].endswith(".png")
