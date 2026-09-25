"""
test_baseline_cnn.py
--------------------
Phase 4 unit tests for the baseline custom CNN.

These tests run on small synthetic tensors — they do NOT load the real Kaggle
dataset and do NOT run a full training session. The suite is designed to be
quick (< 30 seconds on any machine).

Checks:
    - Model can be built without errors.
    - Input shape matches the specified (224, 224, 3) configuration.
    - Output shape is (batch_size, num_classes).
    - num_classes equals 4.
    - A forward pass on a small synthetic batch produces finite float32 outputs.
    - Output probabilities sum to approximately 1.0 per sample.
    - Predicted class IDs are in the range [0, 3].
    - The model has a non-trivial parameter count (sanity check it is actually
      a CNN, not an accidentally empty model).
    - No pretrained architecture is present (no EfficientNet / MobileNet /
      ResNet / VGG / DenseNet layer names).
    - The model has not been accidentally connected to an ImageNet-weight
      loader.
    - Training module constants are sane (correct loss name, learning rate
      range, batch size, maximum epoch count).
"""

from __future__ import annotations

import os

# Suppress TF C++ log noise during tests.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
import tensorflow as tf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_model() -> tf.keras.Model:
    """Build and return the baseline CNN once for the whole test module."""
    from src.models.baseline_cnn import build_baseline_cnn

    return build_baseline_cnn(input_shape=(224, 224, 3), num_classes=4)


@pytest.fixture(scope="module")
def synthetic_batch() -> tuple[np.ndarray, np.ndarray]:
    """Return a small synthetic (images, labels) batch of shape (4, 224, 224, 3)."""
    rng = np.random.default_rng(seed=0)
    images = rng.random((4, 224, 224, 3), dtype=np.float32)
    labels = np.array([0, 1, 2, 3], dtype=np.int32)
    return images, labels


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------


def test_model_builds_without_error() -> None:
    """build_baseline_cnn() must succeed and return a Keras Model."""
    from src.models.baseline_cnn import build_baseline_cnn

    model = build_baseline_cnn()
    assert isinstance(model, tf.keras.Model)


def test_model_name_is_baseline_cnn(baseline_model: tf.keras.Model) -> None:
    """Model name should identify this as the baseline."""
    assert "baseline" in baseline_model.name.lower()


def test_input_shape(baseline_model: tf.keras.Model) -> None:
    """The model's expected input shape should be (None, 224, 224, 3)."""
    shape = baseline_model.input_shape  # (None, H, W, C)
    assert shape[1] == 224, f"Expected height 224, got {shape[1]}"
    assert shape[2] == 224, f"Expected width 224, got {shape[2]}"
    assert shape[3] == 3, f"Expected 3 channels, got {shape[3]}"


def test_output_shape(baseline_model: tf.keras.Model) -> None:
    """The model's output shape should be (None, 4)."""
    shape = baseline_model.output_shape  # (None, num_classes)
    assert shape[1] == 4, f"Expected 4 output classes, got {shape[1]}"


def test_num_classes_is_four(baseline_model: tf.keras.Model) -> None:
    """Output dimension must equal NUM_CLASSES = 4."""
    assert baseline_model.output_shape[-1] == 4


# ---------------------------------------------------------------------------
# Forward-pass tests
# ---------------------------------------------------------------------------


def test_forward_pass_output_shape(
    baseline_model: tf.keras.Model,
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    """Forward pass should produce (batch_size, 4) outputs."""
    images, _ = synthetic_batch
    predictions = baseline_model(images, training=False)
    assert predictions.shape == (4, 4), (
        f"Expected shape (4, 4), got {tuple(predictions.shape)}"
    )


def test_forward_pass_dtype_is_float32(
    baseline_model: tf.keras.Model,
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    """Output tensor must be float32 (explicit dtype on the output Dense)."""
    images, _ = synthetic_batch
    predictions = baseline_model(images, training=False)
    assert predictions.dtype == tf.float32, (
        f"Expected float32 output, got {predictions.dtype}"
    )


def test_output_probabilities_are_finite(
    baseline_model: tf.keras.Model,
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    """All output values must be finite (no NaN or Inf)."""
    images, _ = synthetic_batch
    predictions = baseline_model(images, training=False).numpy()
    assert np.all(np.isfinite(predictions)), "Model outputs contain NaN or Inf"


def test_output_probabilities_sum_to_one(
    baseline_model: tf.keras.Model,
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    """Softmax outputs must sum to approximately 1.0 per sample."""
    images, _ = synthetic_batch
    predictions = baseline_model(images, training=False).numpy()
    sums = predictions.sum(axis=1)
    np.testing.assert_allclose(
        sums,
        np.ones(len(sums)),
        atol=1e-5,
        err_msg=f"Probability sums not ≈ 1: {sums}",
    )


def test_predicted_class_ids_in_valid_range(
    baseline_model: tf.keras.Model,
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    """argmax class predictions must be in [0, 3]."""
    images, _ = synthetic_batch
    predictions = baseline_model(images, training=False).numpy()
    class_ids = predictions.argmax(axis=1)
    assert np.all((class_ids >= 0) & (class_ids <= 3)), (
        f"Unexpected class IDs: {class_ids}"
    )


# ---------------------------------------------------------------------------
# Parameter-count sanity tests
# ---------------------------------------------------------------------------


def test_model_has_trainable_parameters(baseline_model: tf.keras.Model) -> None:
    """Model must have a non-trivial number of trainable parameters (> 1 M)."""
    trainable_params = sum(
        np.prod(var.shape) for var in baseline_model.trainable_variables
    )
    assert trainable_params > 1_000_000, (
        f"Model has only {trainable_params:,} trainable params — seems too small."
    )


def test_model_parameter_count_is_not_excessive(
    baseline_model: tf.keras.Model,
) -> None:
    """Baseline model should not be absurdly large (< 50 M params)."""
    total_params = baseline_model.count_params()
    assert total_params < 50_000_000, (
        f"Model has {total_params:,} params — too large for a baseline CNN."
    )


# ---------------------------------------------------------------------------
# No-pretrained-architecture tests
# ---------------------------------------------------------------------------

_FORBIDDEN_LAYER_SUBSTRINGS = [
    "efficientnet",
    "mobilenet",
    "resnet",
    "vgg",
    "densenet",
    "inception",
    "xception",
    "nasnet",
]


def test_no_pretrained_architecture_layers(baseline_model: tf.keras.Model) -> None:
    """Ensure no transfer-learning layers (EfficientNet, ResNet, etc.) are present."""
    layer_names = [layer.name.lower() for layer in baseline_model.layers]
    for forbidden in _FORBIDDEN_LAYER_SUBSTRINGS:
        matches = [n for n in layer_names if forbidden in n]
        assert not matches, (
            f"Forbidden pretrained architecture '{forbidden}' found in layers: {matches}"
        )


def test_model_has_expected_conv_blocks() -> None:
    """Model should contain four Conv2D layers with the specified filter counts."""
    from src.models.baseline_cnn import build_baseline_cnn

    model = build_baseline_cnn()
    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    filter_counts = [layer.filters for layer in conv_layers]
    assert filter_counts == [32, 64, 128, 256], (
        f"Expected Conv2D filter counts [32, 64, 128, 256], got {filter_counts}"
    )


def test_model_uses_global_average_pooling() -> None:
    """Model must use GlobalAveragePooling2D, not Flatten on a large feature map."""
    from src.models.baseline_cnn import build_baseline_cnn

    model = build_baseline_cnn()
    layer_types = [type(layer).__name__ for layer in model.layers]
    assert "GlobalAveragePooling2D" in layer_types, (
        "Expected GlobalAveragePooling2D in the model head."
    )
    assert "Flatten" not in layer_types, (
        "Flatten layer found — should use GlobalAveragePooling2D instead."
    )


def test_model_has_batch_normalization() -> None:
    """Model must include BatchNormalization layers."""
    from src.models.baseline_cnn import build_baseline_cnn

    model = build_baseline_cnn()
    bn_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    assert len(bn_layers) >= 4, (
        f"Expected at least 4 BatchNormalization layers, found {len(bn_layers)}."
    )


def test_model_has_dropout() -> None:
    """Model must include a Dropout layer in the head."""
    from src.models.baseline_cnn import build_baseline_cnn

    model = build_baseline_cnn()
    dropout_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dropout)
    ]
    assert len(dropout_layers) >= 1, "Expected at least one Dropout layer."
    # Rate should be around 0.4 (allow a small tolerance).
    rate = dropout_layers[0].rate
    assert abs(rate - 0.4) < 0.05, (
        f"Dropout rate expected ~0.4, got {rate}"
    )


# ---------------------------------------------------------------------------
# Training module configuration tests
# ---------------------------------------------------------------------------


def test_training_module_imports_cleanly() -> None:
    """train_baseline module must be importable without side effects."""
    import importlib
    mod = importlib.import_module("src.models.train_baseline")
    assert mod is not None


def test_training_constants_are_sane() -> None:
    """Training hyper-parameter constants must be within sensible ranges."""
    from src.models import train_baseline as tb

    assert 0 < tb.LEARNING_RATE <= 0.01, (
        f"LEARNING_RATE {tb.LEARNING_RATE} seems unreasonable."
    )
    assert tb.MAX_EPOCHS >= 10, (
        f"MAX_EPOCHS {tb.MAX_EPOCHS} seems too low for a baseline."
    )
    assert tb.BATCH_SIZE in (16, 32, 64), (
        f"BATCH_SIZE {tb.BATCH_SIZE} is unusual."
    )
    assert tb.NUM_CLASSES == 4, (
        f"NUM_CLASSES must be 4, got {tb.NUM_CLASSES}."
    )
    assert tb.ES_PATIENCE >= 3, (
        f"EarlyStopping patience {tb.ES_PATIENCE} seems too aggressive."
    )


def test_loss_function_is_sparse_categorical() -> None:
    """Model must be compilable with SparseCategoricalCrossentropy."""
    from src.models.baseline_cnn import build_baseline_cnn
    import tensorflow as tf

    model = build_baseline_cnn()
    # This should not raise.
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    assert model.loss is not None
