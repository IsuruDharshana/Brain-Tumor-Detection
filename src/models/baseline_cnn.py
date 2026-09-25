"""
baseline_cnn.py
---------------
Phase 4 baseline custom CNN for the Brain Tumor MRI Classification project.

This module defines a clean, custom convolutional neural network that serves
as a reproducible baseline before EfficientNetB0 transfer learning in Phase 5.
The architecture is intentionally straightforward and undergraduate-friendly:

    Four convolutional blocks (Conv2D → BatchNorm → MaxPool or just MaxPool
    omitted on the last block to preserve spatial resolution a little longer)
    followed by GlobalAveragePooling2D → Dense(128) → Dropout → Dense(4,
    softmax).

Design choices
--------------
* **GlobalAveragePooling2D** instead of Flatten to avoid a huge dense layer
  after a 7×7×256 feature map and to add mild spatial regularisation.
* **BatchNormalization** after every Conv2D to stabilise gradients and speed
  up convergence without needing a very small learning rate.
* **No pretrained weights** — this is a pure baseline for scientific comparison.
* **float32 output** — the final Dense is cast to float32 explicitly so the
  model stays numerically stable even if mixed precision is tried later.

Public API
----------
    build_baseline_cnn(input_shape, num_classes) -> tf.keras.Model
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
) -> tf.keras.Model:
    """Build and return the Phase 4 baseline custom CNN.

    The model is built with the Keras Functional API so that layer output
    shapes are fully known at build time (useful for ``model.summary()`` and
    for inspecting intermediate activations later).

    Architecture overview::

        Input (224, 224, 3)
        │
        ├─ Block 1: Conv2D(32, 3, same, relu) → BN → MaxPool2D → (112, 112, 32)
        ├─ Block 2: Conv2D(64, 3, same, relu) → BN → MaxPool2D → (56, 56, 64)
        ├─ Block 3: Conv2D(128, 3, same, relu) → BN → MaxPool2D → (28, 28, 128)
        ├─ Block 4: Conv2D(256, 3, same, relu) → BN           → (28, 28, 256)
        │           (no MaxPool: keeps more spatial detail for the GAP)
        │
        ├─ GlobalAveragePooling2D → (256,)
        ├─ Dense(128, relu)
        ├─ Dropout(0.4)
        └─ Dense(num_classes, softmax, dtype=float32) → (4,)

    Args:
        input_shape: (height, width, channels). Defaults to (224, 224, 3).
        num_classes: number of output classes. Defaults to 4.

    Returns:
        A compiled-ready ``tf.keras.Model`` instance (not yet compiled).
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # ------------------------------------------------------------------
    # Block 1 — 32 filters, spatial output: 112 × 112
    # ------------------------------------------------------------------
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="block1_conv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="block1_bn")(x)
    x = tf.keras.layers.MaxPooling2D(name="block1_pool")(x)

    # ------------------------------------------------------------------
    # Block 2 — 64 filters, spatial output: 56 × 56
    # ------------------------------------------------------------------
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="block2_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block2_bn")(x)
    x = tf.keras.layers.MaxPooling2D(name="block2_pool")(x)

    # ------------------------------------------------------------------
    # Block 3 — 128 filters, spatial output: 28 × 28
    # ------------------------------------------------------------------
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="block3_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block3_bn")(x)
    x = tf.keras.layers.MaxPooling2D(name="block3_pool")(x)

    # ------------------------------------------------------------------
    # Block 4 — 256 filters, spatial output: 28 × 28 (no MaxPool)
    # Retaining the 28×28 spatial grid here gives GlobalAveragePooling2D
    # more spatial information to average over, acting as a mild spatial
    # regulariser and reducing parameter count compared to Flatten.
    # ------------------------------------------------------------------
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="block4_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="block4_bn")(x)

    # ------------------------------------------------------------------
    # Head — pooling → dense → dropout → softmax
    # ------------------------------------------------------------------
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(rate=0.4, name="dropout")(x)

    # dtype="float32" ensures the output stays in full precision even if the
    # rest of the model is later run under mixed precision.
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="output_softmax",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    return model
