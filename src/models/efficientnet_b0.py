"""
efficientnet_b0.py
------------------
Phase 5 EfficientNetB0 transfer-learning model for the Brain Tumor MRI
Classification project.

Preprocessing strategy
----------------------
The Phase 3 tf.data pipeline (``src.data.data_pipeline``) supplies images as
**float32 tensors in [0, 1]** (pixel / 255).

``tf.keras.applications.EfficientNetB0`` (TensorFlow/Keras >= 2.x) includes
an *internal* ``Rescaling(1/255)`` layer as the very first layer of the
backbone — i.e. the backbone expects raw **[0, 255]** pixel values and
rescales them itself.

Supplying [0, 1] values directly would apply the 1/255 rescaling a second
time, yielding activations ~255x smaller than at ImageNet training time and
severely degrading feature quality.

Chosen solution — **model-side adapter**:

    Input [0, 1]  (Phase 3 pipeline output)
        |
        v  Rescaling(scale=255.0)   <- adapter layer, part of this model
        |
        v  [0, 255] float32
    EfficientNetB0(include_top=False)
        |  <- backbone's own Rescaling(1/255) normalises to [0,1] internally
        v
    GlobalAveragePooling2D
        v
    Dropout(0.3)
        v
    Dense(128, relu)
        v
    Dropout(0.3)
        v
    Dense(4, softmax, dtype=float32)

This keeps ``src.data.data_pipeline`` **completely unchanged** while
correctly supplying EfficientNetB0 with the value range it was pretrained on.

Note: ``tf.keras.applications.efficientnet.preprocess_input`` is a
pass-through (identity) for the TF/Keras implementation that already contains
the internal Rescaling layer and is therefore NOT called here.

Structural note — why backbone(x) instead of input_tensor=x
-------------------------------------------------------------
Passing ``input_tensor=x`` to ``EfficientNetB0`` inlines all backbone layers
*directly* into the outer Functional graph.  The backbone object becomes a
transient construction helper that is never inserted into ``model.layers``
as a nested ``tf.keras.Model``.  The stage-control helpers
(``freeze_backbone``, ``unfreeze_top_layers``, ``count_trainable_layers``)
rely on finding the backbone as a nested sub-model via::

    isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower()

The correct pattern: build the backbone **standalone** (``input_shape`` only,
no ``input_tensor``), then *call* it as a layer (``backbone(x)``).  Keras
then records it as a single nested ``tf.keras.Model`` entry inside
``model.layers``.

Public API
----------
    build_efficientnet_b0(
        input_shape=(224, 224, 3),
        num_classes=4,
        weights="imagenet",
        dropout_rate=0.3,
    ) -> tf.keras.Model

    freeze_backbone(model) -> None
    unfreeze_top_layers(model, num_layers=30, keep_bn_frozen=True) -> None
    count_trainable_layers(model) -> dict[str, int]
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Substring used to identify the EfficientNetB0 backbone sub-model inside
#: model.layers.  _get_backbone() matches isinstance(layer, tf.keras.Model)
#: AND this substring so the lookup is robust to Keras auto-suffixing
#: (e.g. "efficientnetb0" vs "efficientnetb0_1").
_BACKBONE_SUBSTRING: str = "efficientnet"


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_efficientnet_b0(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    weights: str | None = "imagenet",
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """Build the EfficientNetB0 transfer-learning model.

    The outer Keras Functional model has the following structure::

        inputs          Input(224, 224, 3)          [0, 1] float32
        x               Rescaling(255.0)             [0, 255] float32
        x               EfficientNetB0 (nested)     nested tf.keras.Model
        x               GlobalAveragePooling2D
        x               Dropout(dropout_rate)
        x               Dense(128, relu)
        x               Dropout(dropout_rate)
        outputs         Dense(num_classes, softmax, float32)

    EfficientNetB0 is constructed as a standalone model (input_shape only,
    NO input_tensor) and then *called* as a sub-layer.  This guarantees it
    appears in ``model.layers`` as a nested ``tf.keras.Model``, which is
    required by ``freeze_backbone`` and ``unfreeze_top_layers``.

    Args:
        input_shape: (height, width, channels). Must be (224, 224, 3).
        num_classes: number of output classes. 4 for this project.
        weights: ``"imagenet"`` loads pretrained weights (internet required
            on first call); ``None`` uses random initialisation (for tests).
        dropout_rate: dropout probability in the classifier head.

    Returns:
        An uncompiled ``tf.keras.Model`` with EfficientNetB0 as a nested
        sub-model accessible via ``model.layers``.
    """
    # ------------------------------------------------------------------
    # 1. Outer Input + preprocessing adapter
    # ------------------------------------------------------------------
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Phase 3 pipeline delivers [0, 1]; EfficientNetB0 expects [0, 255].
    x = tf.keras.layers.Rescaling(
        scale=255.0,
        name="adapter_rescale_to_255",
    )(inputs)

    # ------------------------------------------------------------------
    # 2. EfficientNetB0 backbone — STANDALONE, then called as a layer
    #
    # DO NOT pass input_tensor=x here.  Doing so inlines all backbone
    # layers into the outer graph so the backbone never appears as a
    # nested sub-model in model.layers, breaking freeze_backbone().
    # ------------------------------------------------------------------
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,   # standalone: defines backbone's own Input
    )

    # Call backbone as a layer.  Keras records it as a nested tf.keras.Model
    # in model.layers.  training=False pins BatchNorm to stored statistics
    # when building; model.fit() sets this correctly during actual training.
    x = backbone(x, training=False)

    # ------------------------------------------------------------------
    # 3. Classification head
    # ------------------------------------------------------------------
    x = tf.keras.layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense_128")(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate, name="head_dropout_2")(x)

    # Explicit float32 dtype keeps outputs stable regardless of compute dtype.
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="output_softmax",
    )(x)

    # Use `inputs` (our own Input layer) — NOT backbone.input — so the
    # Rescaling adapter sits inside the outer model graph.
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="efficientnet_b0_transfer",
    )
    return model


# ---------------------------------------------------------------------------
# Stage-control helpers
# ---------------------------------------------------------------------------

def _get_backbone(model: tf.keras.Model) -> tf.keras.Model | None:
    """Return the EfficientNetB0 nested sub-model, or None if not found.

    Searches ``model.layers`` for an entry that is itself a
    ``tf.keras.Model`` and whose name contains "efficientnet".  The
    isinstance check is more reliable than a name-only match because Keras
    may append version suffixes (e.g. "efficientnetb0_1").
    """
    for layer in model.layers:
        if (
            isinstance(layer, tf.keras.Model)
            and _BACKBONE_SUBSTRING in layer.name.lower()
        ):
            return layer
    return None


def freeze_backbone(model: tf.keras.Model) -> None:
    """Freeze every layer of the EfficientNetB0 backbone (Stage 1).

    After calling this only the custom classifier head is trainable.
    BatchNormalization layers inside the backbone are frozen too — they
    run in inference mode, using ImageNet-learned statistics rather than
    accumulating new statistics from the small MRI dataset.

    Args:
        model: full EfficientNetB0 transfer model from ``build_efficientnet_b0``.

    Raises:
        RuntimeError: if the backbone sub-model cannot be located.
    """
    backbone = _get_backbone(model)
    if backbone is None:
        raise RuntimeError(
            "Could not find EfficientNetB0 as a nested sub-model in model.layers. "
            "Ensure the model was built with build_efficientnet_b0() and that "
            "input_tensor was NOT passed to EfficientNetB0()."
        )
    backbone.trainable = False


def unfreeze_top_layers(
    model: tf.keras.Model,
    num_layers: int = 30,
    keep_bn_frozen: bool = True,
) -> None:
    """Unfreeze only the last ``num_layers`` backbone layers for Stage 2.

    Strategy:
        1. Set backbone.trainable = True (makes all backbone layers trainable).
        2. Re-freeze all backbone layers except the last ``num_layers``.
        3. Optionally re-freeze all BatchNormalization layers (recommended
           for small datasets to preserve ImageNet statistics).

    The classifier head layers are already trainable from Stage 1 and are
    not touched by this function.

    Args:
        model: full transfer model from ``build_efficientnet_b0``.
        num_layers: number of top backbone layers to unfreeze (default 30).
        keep_bn_frozen: if True, BatchNorm layers stay frozen even in the
            unfrozen region.

    Raises:
        RuntimeError: if the backbone sub-model cannot be located.
    """
    backbone = _get_backbone(model)
    if backbone is None:
        raise RuntimeError(
            "Could not find EfficientNetB0 as a nested sub-model in model.layers."
        )

    # Step 1 — unfreeze the whole backbone
    backbone.trainable = True

    # Step 2 — re-freeze everything below the top num_layers
    freeze_up_to = max(0, len(backbone.layers) - num_layers)
    for layer in backbone.layers[:freeze_up_to]:
        layer.trainable = False

    # Step 3 — optionally keep all BN layers frozen
    if keep_bn_frozen:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


def count_trainable_layers(model: tf.keras.Model) -> dict[str, int]:
    """Return trainable/frozen layer counts across the full model.

    Descends into nested sub-models (the backbone) to count individual
    component layers rather than treating the backbone as a single unit.

    Returns:
        dict with keys ``"total"``, ``"trainable"``, ``"frozen"``.
    """
    flat_layers: list[tf.keras.layers.Layer] = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # nested backbone → descend
            flat_layers.extend(layer.layers)
        else:
            flat_layers.append(layer)

    trainable = sum(1 for ll in flat_layers if ll.trainable)
    return {
        "total": len(flat_layers),
        "trainable": trainable,
        "frozen": len(flat_layers) - trainable,
    }
