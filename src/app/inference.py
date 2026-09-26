"""
inference.py
------------
Reusable preprocessing and inference pipeline for the Phase 7 Streamlit app.

Independent of Streamlit for easy testing and integration.
Operates entirely in-memory with no disk writes.

Model assumptions:
  - Baseline CNN (models/baseline_cnn.keras)
  - Input: (1, 224, 224, 3) float32 in [0, 1]
  - Classes: 0=glioma, 1=meningioma, 2=notumor, 3=pituitary
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import CLASS_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH
from src.data.preprocessing import decode_and_preprocess_image_bytes, load_and_preprocess_image

CLASS_DISPLAY_NAMES: dict[str, str] = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

# Conservative thresholds for rejecting strongly colourful inputs. All colour
# metrics are normalized to [0, 1]. These values deliberately allow mild tinting.
MAX_MEAN_SATURATION = 0.30
MAX_P95_SATURATION = 0.65
MAX_MEAN_CHANNEL_DIFFERENCE = 0.12
MAX_COLORFUL_PIXEL_FRACTION = 0.35
COLORFUL_PIXEL_SATURATION = 0.50
COLORFUL_PIXEL_CHANNEL_DIFFERENCE = 0.15
MIN_INTENSITY_STD = 0.01
MIN_INTENSITY_RANGE = 0.02
VALIDATION_MAX_SIDE = 512


def validate_mri_like_image(image: Image.Image) -> dict[str, Any]:
    """Apply a conservative grayscale-MRI plausibility check to a PIL image.

    This is a heuristic plausibility filter intended to reject obviously
    non-MRI inputs. It is not an out-of-distribution detector and does not
    verify that an image is a genuine medical MRI.

    Strong colour is detected using HSV-equivalent saturation, RGB channel
    spread, and the fraction of clearly colourful pixels. A secondary check
    rejects images that are effectively uniform or blank. Mild tinting is
    intentionally allowed.

    Args:
        image: Decoded PIL image, before model resizing and normalization.

    Returns:
        A dictionary containing ``is_plausible_mri``, a human-readable
        ``reason``, and normalized image-level ``metrics`` for tests/debugging.

    Raises:
        ValueError: if ``image`` is not a non-empty PIL image.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("MRI plausibility validation requires a PIL image.")
    if image.width < 1 or image.height < 1:
        raise ValueError("MRI plausibility validation requires a non-empty image.")

    # Bound validation work for very large uploads without changing the image
    # passed to the existing model preprocessing pipeline.
    validation_image = image.convert("RGB")
    if max(validation_image.size) > VALIDATION_MAX_SIDE:
        validation_image = validation_image.copy()
        resample_filter = getattr(
            getattr(Image, "Resampling", Image),
            "BILINEAR",
            Image.BILINEAR,
        )
        validation_image.thumbnail(
            (VALIDATION_MAX_SIDE, VALIDATION_MAX_SIDE),
            resample=resample_filter,
        )

    rgb = np.asarray(validation_image, dtype=np.float32) / 255.0
    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    channel_spread = max_channel - min_channel

    saturation = np.zeros_like(max_channel, dtype=np.float32)
    np.divide(
        channel_spread,
        max_channel,
        out=saturation,
        where=max_channel > 1e-6,
    )

    colorful_pixels = (
        (saturation >= COLORFUL_PIXEL_SATURATION)
        & (channel_spread >= COLORFUL_PIXEL_CHANNEL_DIFFERENCE)
    )

    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )

    metrics = {
        "mean_saturation": float(np.mean(saturation)),
        "p95_saturation": float(np.percentile(saturation, 95)),
        "mean_channel_difference": float(np.mean(channel_spread)),
        "colorful_pixel_fraction": float(np.mean(colorful_pixels)),
        "intensity_std": float(np.std(luminance)),
        "intensity_range": float(
            np.percentile(luminance, 99) - np.percentile(luminance, 1)
        ),
    }

    strongly_colorful = (
        metrics["mean_saturation"] >= MAX_MEAN_SATURATION
        and metrics["p95_saturation"] >= MAX_P95_SATURATION
        and metrics["mean_channel_difference"] >= MAX_MEAN_CHANNEL_DIFFERENCE
    )
    broadly_colorful = (
        metrics["colorful_pixel_fraction"] >= MAX_COLORFUL_PIXEL_FRACTION
        and metrics["mean_channel_difference"] >= MAX_MEAN_CHANNEL_DIFFERENCE
    )

    if strongly_colorful or broadly_colorful:
        return {
            "is_plausible_mri": False,
            "reason": "The image contains strong colour patterns unlike a grayscale MRI.",
            "metrics": metrics,
        }

    is_blank_or_uniform = (
        metrics["intensity_std"] < MIN_INTENSITY_STD
        and metrics["intensity_range"] < MIN_INTENSITY_RANGE
    )
    if is_blank_or_uniform:
        return {
            "is_plausible_mri": False,
            "reason": "The image is blank or too uniform for meaningful analysis.",
            "metrics": metrics,
        }

    return {
        "is_plausible_mri": True,
        "reason": "The image passed the basic grayscale-MRI plausibility check.",
        "metrics": metrics,
    }


def preprocess_image(
    image_input: Image.Image | bytes | bytearray | io.BytesIO | str | Path,
    target_size: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
) -> np.ndarray:
    """Validate and preprocess an input image for model inference.

    Uses the exact TensorFlow preprocessing pipeline from Phase 3 to guarantee
    numerical reproducibility with training and Phase 6 evaluation.

    Accepts file bytes, bytearray, BytesIO streams, file paths, or PIL Images.
    For bytes / bytearray / BytesIO:
        Preprocesses original encoded file bytes directly via
        decode_and_preprocess_image_bytes().
    For file paths:
        Preprocesses via load_and_preprocess_image().
    For PIL Images (supported for testing/internal callers):
        Converts to uint8 RGB tensor, then applies convert_image_dtype to float32
        and tf.image.resize using TensorFlow's bilinear interpolation.

    Args:
        image_input: image in any supported format.
        target_size: (height, width) dimensions expected by the model.

    Returns:
        np.ndarray of shape (1, height, width, 3) and dtype float32 with
        values in [0.0, 1.0].

    Raises:
        ValueError: if image_input cannot be decoded or is invalid.
    """
    if isinstance(image_input, (bytes, bytearray, io.BytesIO)):
        if isinstance(image_input, io.BytesIO):
            raw_bytes = image_input.getvalue()
        elif isinstance(image_input, (bytearray, memoryview)):
            raw_bytes = bytes(image_input)
        else:
            raw_bytes = image_input

        if not raw_bytes:
            raise ValueError("Empty image bytes provided.")

        try:
            tensor = decode_and_preprocess_image_bytes(raw_bytes)
            if target_size != (IMAGE_HEIGHT, IMAGE_WIDTH):
                tensor = tf.image.resize(tensor, [target_size[0], target_size[1]])
            arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"Could not decode image from provided bytes: {exc}") from exc

    elif isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise ValueError(f"Image file does not exist: {p}")
        try:
            tensor = load_and_preprocess_image(str(p))
            if target_size != (IMAGE_HEIGHT, IMAGE_WIDTH):
                tensor = tf.image.resize(tensor, [target_size[0], target_size[1]])
            arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"Could not load and preprocess image file {p}: {exc}") from exc

    elif isinstance(image_input, Image.Image):
        if image_input.width < 1 or image_input.height < 1:
            raise ValueError("Cannot preprocess an empty image.")
        try:
            pil_img = image_input
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            np_img = np.asarray(pil_img, dtype=np.uint8)
            tf_img = tf.convert_to_tensor(np_img, dtype=tf.uint8)
            tf_img.set_shape([None, None, 3])
            tf_img = tf.image.convert_image_dtype(tf_img, tf.float32)
            tensor = tf.image.resize(tf_img, [target_size[0], target_size[1]])
            arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"Could not preprocess PIL image: {exc}") from exc

    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Ensure batch dimension (1, 224, 224, 3)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    return arr


def predict_image(model: Any, preprocessed_image: np.ndarray) -> dict[str, Any]:
    """Run model inference on a preprocessed image array and return structured predictions.

    Args:
        model: loaded Keras model (or callable mock).
        preprocessed_image: numpy array of shape (1, 224, 224, 3) or (224, 224, 3)
            with values in [0.0, 1.0].

    Returns:
        dict containing:
            predicted_class_id: int (0 to 3)
            predicted_class_name: str ("glioma", "meningioma", "notumor", "pituitary")
            predicted_display_name: str ("Glioma", "Meningioma", "No Tumor", "Pituitary")
            confidence: float (max probability)
            probabilities: dict[str, float] mapping canonical class names to probabilities
            display_probabilities: dict[str, float] mapping UI display names to probabilities

    Raises:
        ValueError: if input array is invalid, model output shape is unexpected,
            or output values contain NaN/infinite/invalid probabilities.
    """
    if not isinstance(preprocessed_image, np.ndarray):
        preprocessed_image = np.asarray(preprocessed_image, dtype=np.float32)

    if preprocessed_image.ndim == 3:
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    if preprocessed_image.ndim != 4 or preprocessed_image.shape[-1] != 3:
        raise ValueError(
            f"Expected input array of shape (1, H, W, 3), got shape {preprocessed_image.shape}"
        )

    # Execute model inference
    try:
        # Prefer callable model(x, training=False) if available, fallback to predict()
        if callable(model):
            try:
                preds = model(preprocessed_image, training=False)
            except TypeError:
                preds = model(preprocessed_image)
        elif hasattr(model, "predict"):
            preds = model.predict(preprocessed_image, verbose=0)
        else:
            raise TypeError("Model object is neither callable nor possesses a predict method.")
    except Exception as exc:
        raise RuntimeError(f"Error during model prediction: {exc}") from exc

    # Convert tensor to numpy array
    if hasattr(preds, "numpy"):
        probs = preds.numpy()
    else:
        probs = np.asarray(preds, dtype=np.float32)

    if probs.ndim == 2:
        if probs.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 output, got batch size {probs.shape[0]}")
        probs = probs[0]

    num_classes = len(CLASS_NAMES)
    if probs.shape != (num_classes,):
        raise ValueError(
            f"Expected model output of shape ({num_classes},), got {probs.shape}"
        )

    # Validate numeric integrity
    if not np.all(np.isfinite(probs)):
        raise ValueError("Model output contains NaN or infinite values.")

    if np.any(probs < -1e-4) or np.any(probs > 1.0 + 1e-4):
        raise ValueError(
            f"Model output contains values outside valid probability range [0, 1]: {probs}"
        )

    # Clip to valid [0, 1] range to safeguard against tiny precision artifacts
    probs = np.clip(probs, 0.0, 1.0)

    pred_id = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_id]
    confidence = float(probs[pred_id])

    prob_dict = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
    display_prob_dict = {
        CLASS_DISPLAY_NAMES[name]: float(probs[i]) for i, name in enumerate(CLASS_NAMES)
    }

    return {
        "predicted_class_id": pred_id,
        "predicted_class_name": pred_name,
        "predicted_display_name": CLASS_DISPLAY_NAMES[pred_name],
        "confidence": confidence,
        "probabilities": prob_dict,
        "display_probabilities": display_prob_dict,
    }
