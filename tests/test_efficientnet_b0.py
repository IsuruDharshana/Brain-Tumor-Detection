"""
test_efficientnet_b0.py
-----------------------
Phase 5 unit tests for the EfficientNetB0 transfer-learning model.

All tests run on small synthetic tensors with ``weights=None`` so that:
- No internet access is required (no ImageNet weight download).
- Tests complete quickly (< 60 seconds even on CPU).
- No real Kaggle dataset is needed.

Test coverage:
    Architecture:
        - Model builds without error (weights=None)
        - Correct input shape (224, 224, 3)
        - Correct output shape (batch, 4)
        - Four output classes
        - Output dtype is float32
        - EfficientNetB0 backbone is present as a sub-layer
        - GlobalAveragePooling2D is used in the head
        - No Flatten layer in the classifier head
        - Dropout layers present in the head
        - Rescaling(255.0) adapter layer exists in the model graph
        - No Flatten layer anywhere

    Forward pass:
        - Produces finite outputs
        - Softmax probabilities sum to ~1
        - Predicted class IDs in [0, 3]
        - Works on [0, 1] float32 input (Phase 3 pipeline range)

    Backbone control:
        - freeze_backbone makes backbone non-trainable
        - unfreeze_top_layers restores some trainability
        - count_trainable_layers returns sensible counts

    Training module:
        - train_efficientnet imports cleanly (no side effects)
        - Training constants are reasonable

    Preprocessing:
        - Model adapter layer exists and has scale=255.0
        - Forward pass with [0,1] input produces valid softmax output
"""

from __future__ import annotations

import os

# Suppress TF log noise during tests.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
import tensorflow as tf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eff_model() -> tf.keras.Model:
    """Build EfficientNetB0 with weights=None (no ImageNet download)."""
    from src.models.efficientnet_b0 import build_efficientnet_b0
    return build_efficientnet_b0(
        input_shape=(224, 224, 3),
        num_classes=4,
        weights=None,
        dropout_rate=0.3,
    )


@pytest.fixture(scope="module")
def synthetic_batch() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic batch in [0, 1] float32 — exactly what Phase 3 delivers."""
    rng = np.random.default_rng(seed=7)
    images = rng.random((4, 224, 224, 3), dtype=np.float32)
    labels = np.array([0, 1, 2, 3], dtype=np.int32)
    return images, labels


# ---------------------------------------------------------------------------
# Architecture — build
# ---------------------------------------------------------------------------

def test_model_builds_without_error() -> None:
    """build_efficientnet_b0(weights=None) must succeed."""
    from src.models.efficientnet_b0 import build_efficientnet_b0
    model = build_efficientnet_b0(weights=None)
    assert isinstance(model, tf.keras.Model)


def test_model_name_contains_efficientnet(eff_model: tf.keras.Model) -> None:
    assert "efficientnet" in eff_model.name.lower()


# ---------------------------------------------------------------------------
# Architecture — shapes
# ---------------------------------------------------------------------------

def test_input_shape(eff_model: tf.keras.Model) -> None:
    """Model input must be (None, 224, 224, 3)."""
    shape = eff_model.input_shape
    assert shape[1] == 224, f"Height: expected 224, got {shape[1]}"
    assert shape[2] == 224, f"Width: expected 224, got {shape[2]}"
    assert shape[3] == 3, f"Channels: expected 3, got {shape[3]}"


def test_output_shape(eff_model: tf.keras.Model) -> None:
    """Model output must be (None, 4)."""
    shape = eff_model.output_shape
    assert shape[-1] == 4, f"Expected 4 classes, got {shape[-1]}"


def test_num_classes_is_four(eff_model: tf.keras.Model) -> None:
    assert eff_model.output_shape[-1] == 4


# ---------------------------------------------------------------------------
# Architecture — layer structure
# ---------------------------------------------------------------------------

def _get_all_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Recursively collect all layers including those inside sub-models."""
    result = []
    for layer in model.layers:
        if hasattr(layer, "layers"):
            result.extend(_get_all_layers(layer))
        else:
            result.append(layer)
    return result


def test_efficientnetb0_backbone_present(eff_model: tf.keras.Model) -> None:
    """EfficientNetB0 must appear as a nested sub-model."""
    sub_model_names = [
        layer.name.lower() for layer in eff_model.layers
        if hasattr(layer, "layers")
    ]
    assert any("efficientnet" in n for n in sub_model_names), (
        f"EfficientNetB0 backbone not found. Sub-models: {sub_model_names}"
    )


def test_global_average_pooling_present(eff_model: tf.keras.Model) -> None:
    """GlobalAveragePooling2D must be used in the head."""
    all_layers = _get_all_layers(eff_model)
    gap_layers = [l for l in all_layers
                  if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)]
    assert len(gap_layers) >= 1, "Expected GlobalAveragePooling2D in the head."


def test_no_flatten_in_head(eff_model: tf.keras.Model) -> None:
    """Flatten must NOT appear in the classifier head (top-level layers)."""
    # Check only the direct/non-backbone layers
    head_layer_types = [
        type(layer).__name__ for layer in eff_model.layers
        if not hasattr(layer, "layers")  # skip sub-models
    ]
    assert "Flatten" not in head_layer_types, (
        "Flatten found in classifier head — should use GlobalAveragePooling2D."
    )


def test_dropout_present_in_head(eff_model: tf.keras.Model) -> None:
    """At least one Dropout layer must exist in the head."""
    head_dropout = [
        layer for layer in eff_model.layers
        if isinstance(layer, tf.keras.layers.Dropout)
    ]
    assert len(head_dropout) >= 1, "Expected at least one Dropout in the head."


def test_output_dtype_is_float32(eff_model: tf.keras.Model) -> None:
    """Output Dense layer must use float32 dtype."""
    output_layer = eff_model.layers[-1]
    # The dtype might be stored on the layer config or the layer compute dtype.
    assert output_layer.dtype == "float32" or output_layer.compute_dtype == "float32", (
        f"Output layer dtype is {output_layer.dtype}, expected float32."
    )


# ---------------------------------------------------------------------------
# Architecture — preprocessing adapter
# ---------------------------------------------------------------------------

def test_rescaling_adapter_exists(eff_model: tf.keras.Model) -> None:
    """A Rescaling(255.0) adapter layer must be present in the model."""
    all_layers = _get_all_layers(eff_model)
    rescaling_layers = [
        l for l in all_layers
        if isinstance(l, tf.keras.layers.Rescaling)
    ]
    assert len(rescaling_layers) >= 1, (
        "No Rescaling layer found. "
        "Expected Rescaling(255.0) adapter before the backbone."
    )


def test_rescaling_adapter_scale_is_255(eff_model: tf.keras.Model) -> None:
    """The adapter Rescaling layer must have scale=255.0."""
    # The adapter is the first Rescaling layer in the top-level graph (before
    # the backbone's own internal layers).
    top_level_rescaling = [
        layer for layer in eff_model.layers
        if isinstance(layer, tf.keras.layers.Rescaling)
    ]
    assert len(top_level_rescaling) >= 1, (
        "Rescaling adapter not found at the top level of the model."
    )
    scale = top_level_rescaling[0].scale
    assert abs(scale - 255.0) < 0.1, (
        f"Adapter Rescaling scale expected 255.0, got {scale}."
    )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_forward_pass_output_shape(
    eff_model: tf.keras.Model,
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    """Forward pass must produce (4, 4) outputs for a 4-sample batch."""
    images, _ = synthetic_batch
    preds = eff_model(images, training=False)
    assert tuple(preds.shape) == (4, 4), (
        f"Expected (4, 4), got {tuple(preds.shape)}"
    )


def test_forward_pass_is_float32(
    eff_model: tf.keras.Model,
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    images, _ = synthetic_batch
    preds = eff_model(images, training=False)
    assert preds.dtype == tf.float32, f"Expected float32, got {preds.dtype}"


def test_probabilities_are_finite(
    eff_model: tf.keras.Model,
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    images, _ = synthetic_batch
    preds = eff_model(images, training=False).numpy()
    assert np.all(np.isfinite(preds)), "Output contains NaN or Inf."


def test_probabilities_sum_to_one(
    eff_model: tf.keras.Model,
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    images, _ = synthetic_batch
    preds = eff_model(images, training=False).numpy()
    sums = preds.sum(axis=1)
    np.testing.assert_allclose(
        sums, np.ones(len(sums)), atol=1e-5,
        err_msg=f"Probability sums: {sums}"
    )


def test_class_ids_in_valid_range(
    eff_model: tf.keras.Model,
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    images, _ = synthetic_batch
    preds = eff_model(images, training=False).numpy()
    class_ids = preds.argmax(axis=1)
    assert np.all((class_ids >= 0) & (class_ids <= 3)), (
        f"Unexpected class IDs: {class_ids}"
    )


def test_forward_pass_accepts_zero_one_input(
    eff_model: tf.keras.Model,
) -> None:
    """Model must correctly process [0,1] float32 inputs (Phase 3 range)."""
    rng = np.random.default_rng(seed=99)
    images = rng.random((2, 224, 224, 3), dtype=np.float32)  # in [0, 1]
    preds = eff_model(images, training=False).numpy()
    assert np.all(np.isfinite(preds)), (
        "Forward pass on [0,1] input produced non-finite outputs."
    )
    sums = preds.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(2), atol=1e-5)


# ---------------------------------------------------------------------------
# Backbone control — freeze / unfreeze
# ---------------------------------------------------------------------------

def test_freeze_backbone_makes_backbone_nontrainable() -> None:
    """freeze_backbone() must freeze all backbone parameters."""
    from src.models.efficientnet_b0 import build_efficientnet_b0, freeze_backbone
    model = build_efficientnet_b0(weights=None)
    freeze_backbone(model)

    # Find the backbone sub-model.
    backbone = next(
        (l for l in model.layers if hasattr(l, "layers") and "efficientnet" in l.name.lower()),
        None,
    )
    assert backbone is not None, "Backbone not found."
    assert not backbone.trainable, "Backbone should be non-trainable after freeze_backbone()."


def test_freeze_backbone_head_remains_trainable() -> None:
    """After freeze_backbone(), the head Dense layers must still be trainable."""
    from src.models.efficientnet_b0 import build_efficientnet_b0, freeze_backbone
    model = build_efficientnet_b0(weights=None)
    freeze_backbone(model)

    head_trainable = any(
        layer.trainable
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    )
    assert head_trainable, "Head Dense layers should remain trainable after freezing backbone."


def test_unfreeze_top_layers_restores_trainability() -> None:
    """unfreeze_top_layers() must make at least some backbone layers trainable."""
    from src.models.efficientnet_b0 import (
        build_efficientnet_b0,
        freeze_backbone,
        unfreeze_top_layers,
        count_trainable_layers,
    )
    model = build_efficientnet_b0(weights=None)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    freeze_backbone(model)
    before = count_trainable_layers(model)

    unfreeze_top_layers(model, num_layers=20, keep_bn_frozen=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    after = count_trainable_layers(model)

    assert after["trainable"] > before["trainable"], (
        f"unfreeze_top_layers() did not increase trainable layer count. "
        f"Before: {before['trainable']}, after: {after['trainable']}"
    )


def test_count_trainable_layers_returns_dict() -> None:
    from src.models.efficientnet_b0 import (
        build_efficientnet_b0,
        count_trainable_layers,
    )
    model = build_efficientnet_b0(weights=None)
    counts = count_trainable_layers(model)
    assert "total" in counts
    assert "trainable" in counts
    assert "frozen" in counts
    assert counts["total"] == counts["trainable"] + counts["frozen"]


def test_stage1_has_fewer_trainable_params_than_stage2() -> None:
    """Stage 1 (frozen) must have fewer trainable params than Stage 2."""
    from src.models.efficientnet_b0 import (
        build_efficientnet_b0,
        freeze_backbone,
        unfreeze_top_layers,
    )
    model = build_efficientnet_b0(weights=None)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    freeze_backbone(model)

    s1_trainable = sum(
        np.prod(v.shape) for v in model.trainable_variables
    )

    unfreeze_top_layers(model, num_layers=30, keep_bn_frozen=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    s2_trainable = sum(
        np.prod(v.shape) for v in model.trainable_variables
    )

    assert s2_trainable > s1_trainable, (
        f"Stage 2 should have more trainable params than Stage 1. "
        f"S1: {s1_trainable:,}, S2: {s2_trainable:,}"
    )


# ---------------------------------------------------------------------------
# Training module sanity
# ---------------------------------------------------------------------------

def test_train_efficientnet_imports_cleanly() -> None:
    """train_efficientnet must be importable without side effects."""
    import importlib
    mod = importlib.import_module("src.models.train_efficientnet")
    assert mod is not None


def test_training_constants_are_reasonable() -> None:
    from src.models import train_efficientnet as te
    assert 0 < te.S1_LR <= 0.01, f"S1_LR {te.S1_LR} unreasonable"
    assert 0 < te.S2_LR < te.S1_LR, (
        f"Fine-tuning LR {te.S2_LR} should be smaller than Stage 1 LR {te.S1_LR}"
    )
    assert te.S1_MAX_EPOCHS >= 5, "Stage 1 max epochs too low"
    assert te.S2_MAX_EPOCHS >= 5, "Stage 2 max epochs too low"
    assert te.BATCH_SIZE in (16, 32, 64), f"BATCH_SIZE {te.BATCH_SIZE} unusual"
    assert te.NUM_CLASSES == 4
    assert te.S2_UNFREEZE_LAYERS > 0, "Must unfreeze at least some layers"
    assert te.S2_UNFREEZE_LAYERS <= 100, "Unfreezing too many layers"


def test_model_is_compilable_with_sparse_crossentropy() -> None:
    """Model must compile without error using SparseCategoricalCrossentropy."""
    from src.models.efficientnet_b0 import build_efficientnet_b0
    model = build_efficientnet_b0(weights=None)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    assert model.loss is not None
