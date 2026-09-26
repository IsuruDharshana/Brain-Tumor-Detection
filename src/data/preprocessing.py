"""
preprocessing.py
----------------
Phase 3 base (deterministic) image preprocessing for the Brain Tumor MRI
Classification project.

This is the ONLY image transformation applied to every split (train,
validation and test), so validation/test data is processed exactly like
training data:

    read file bytes        (tf.io.read_file or raw bytes)
    -> decode              (tf.io.decode_image: detects the real format, so the
                            four PNG-content files carrying a .jpg extension
                            decode correctly)
    -> convert to RGB      (channels=3; L / RGBA / P sources become RGB)
    -> float32 + divide by 255.0  (tf.image.convert_image_dtype -> [0, 1])
    -> resize to 224 x 224 (float32 bilinear)

Random augmentation is NOT part of this module - it lives in
``src.data.data_pipeline.create_training_augmentation`` and is applied only on
top of this base preprocessing for the training split.
"""

from __future__ import annotations

import io
import os

# Keep TensorFlow's C++ log noise low (set before TF is imported).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from src.config import CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH


def decode_and_preprocess_image_bytes(
    file_bytes: tf.Tensor | bytes | bytearray | io.BytesIO,
) -> tf.Tensor:
    """Decode raw image bytes and return a (224, 224, 3) float32 tensor in [0, 1].

    Performs the exact canonical Phase 3 preprocessing pipeline:
        tf.io.decode_image(..., channels=3, expand_animations=False)
        -> set_shape([None, None, 3])
        -> tf.image.convert_image_dtype(..., tf.float32)
        -> tf.image.resize(..., [224, 224])

    Args:
        file_bytes: raw image bytes (bytes, bytearray, io.BytesIO, or tf string tensor).

    Returns:
        float32 tensor of shape (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) with
        pixel values divided by 255.0 into the range [0, 1].
    """
    if isinstance(file_bytes, io.BytesIO):
        file_bytes = file_bytes.getvalue()
    elif isinstance(file_bytes, (bytearray, memoryview)):
        file_bytes = bytes(file_bytes)

    image = tf.io.decode_image(file_bytes, channels=CHANNELS, expand_animations=False)
    # decode_image returns an unknown static shape; set_shape fixes channels to 3.
    image.set_shape([None, None, CHANNELS])
    image = tf.image.convert_image_dtype(image, tf.float32)  # uint8 -> [0, 1]
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return image


def load_and_preprocess_image(path: tf.Tensor | str) -> tf.Tensor:
    """Decode one image file and return a (224, 224, 3) float32 tensor in [0, 1].

    Args:
        path: file path (str, Path or a tf string tensor) of a raw dataset image.

    Returns:
        float32 tensor of shape (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) with
        pixel values divided by 255.0 into the range [0, 1].
    """
    file_bytes = tf.io.read_file(path)
    return decode_and_preprocess_image_bytes(file_bytes)
