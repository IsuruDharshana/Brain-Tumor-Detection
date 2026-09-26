"""
gradcam_service.py
------------------
In-memory Grad-CAM service for the Phase 7 Streamlit application.

Reuses core functionality from src.explainability.gradcam without code duplication.
Operates strictly in memory; no uploaded images or heatmaps are written to disk.
Provides safe error isolation so Grad-CAM issues cannot crash the main application.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.explainability.gradcam import compute_gradcam, find_conv_layer, overlay_heatmap

logger = logging.getLogger(__name__)

DEFAULT_TARGET_LAYER_NAME = "block4_conv"


def generate_gradcam_for_image(
    model: Any,
    preprocessed_image: np.ndarray,
    target_class_id: int | None = None,
    target_layer_name: str = DEFAULT_TARGET_LAYER_NAME,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> dict[str, Any]:
    """Generate in-memory Grad-CAM heatmap and blended overlay for a single image.

    Args:
        model: loaded Keras model (Baseline CNN).
        preprocessed_image: numpy array of shape (1, 224, 224, 3) or (224, 224, 3)
            with values normalized in [0.0, 1.0].
        target_class_id: class index (0-3) to explain. If None, uses top predicted class.
        target_layer_name: preferred convolutional layer name ("block4_conv").
        alpha: heatmap blending factor (0.0 to 1.0).
        colormap: matplotlib colormap name (default "jet").

    Returns:
        dict containing:
            success: bool (True if successfully generated, False otherwise)
            target_layer_name: str (name of the conv layer used)
            original_image: np.ndarray (224, 224, 3) float32 in [0, 1]
            heatmap: np.ndarray (224, 224) float32 in [0, 1] or None
            overlay: np.ndarray (224, 224, 3) float32 in [0, 1] or None
            error_message: str | None (description of failure if success is False)
    """
    img_np = np.asarray(preprocessed_image, dtype=np.float32)
    if img_np.ndim == 4:
        img_rgb = img_np[0]
    elif img_np.ndim == 3:
        img_rgb = img_np
    else:
        return {
            "success": False,
            "target_layer_name": target_layer_name,
            "original_image": None,
            "heatmap": None,
            "overlay": None,
            "error_message": f"Invalid image array dimension: {img_np.ndim}",
        }

    try:
        # Locate target convolutional layer (block4_conv)
        layer = find_conv_layer(model, preferred_name=target_layer_name)
        resolved_layer_name = layer.name

        # Compute Grad-CAM heatmap
        heatmap = compute_gradcam(
            model=model,
            image=img_rgb,
            class_idx=target_class_id,
            layer=layer,
        )

        # Generate blended overlay
        overlay = overlay_heatmap(
            image=img_rgb,
            heatmap=heatmap,
            alpha=alpha,
            colormap=colormap,
        )

        return {
            "success": True,
            "target_layer_name": resolved_layer_name,
            "original_image": img_rgb,
            "heatmap": heatmap,
            "overlay": overlay,
            "error_message": None,
        }

    except Exception as exc:
        logger.warning("Grad-CAM generation failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "target_layer_name": target_layer_name,
            "original_image": img_rgb,
            "heatmap": None,
            "overlay": None,
            "error_message": f"Could not generate Grad-CAM visualization: {exc}",
        }
