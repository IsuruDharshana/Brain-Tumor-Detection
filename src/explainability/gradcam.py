"""
gradcam.py
----------
Phase 6 Grad-CAM explainability utilities for the Brain Tumor MRI
Classification project.

Grad-CAM (Gradient-weighted Class Activation Mapping) visualises which
spatial regions of an MRI image most influenced a model's prediction.

IMPORTANT DISCLAIMER
--------------------
Grad-CAM is an EXPLANATORY VISUALISATION TOOL only.

It highlights image regions that contributed to the model's classification
decision.  It is NOT:
    - tumour segmentation
    - a clinical diagnostic tool
    - proof of lesion location
    - medically validated

This software is educational/research software.

Supported models
----------------
Baseline CNN:
    Flat Functional model with named Conv2D layers.
    Target layer: last Conv2D in the top-level model
    (typically "block4_conv" for the baseline architecture).

EfficientNetB0 transfer model:
    Outer Functional model wrapping a nested EfficientNetB0 sub-model.
    Target layer: last Conv2D inside the backbone sub-model
    (typically "top_conv" or similar).
    Handles the nested model structure automatically.

Public API
----------
    find_conv_layer(model, preferred_name=None) -> tf.keras.layers.Layer
    compute_gradcam(model, image, class_idx=None, layer=None) -> np.ndarray
    overlay_heatmap(image, heatmap, alpha=0.4) -> np.ndarray
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------

def _is_conv_layer(layer: tf.keras.layers.Layer) -> bool:
    """Return True if layer is a Conv2D (or equivalent convolutional layer)."""
    return isinstance(layer, (tf.keras.layers.Conv2D,))


def _collect_conv_layers(
    model: tf.keras.Model,
) -> list[tf.keras.layers.Layer]:
    """Collect all Conv2D layers from a model, descending into sub-models.

    Returns layers in model-forward order (last element = deepest conv
    closest to the output, which is the Grad-CAM target).
    """
    layers: list[tf.keras.layers.Layer] = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            # Nested sub-model (e.g. EfficientNetB0 backbone) — recurse.
            layers.extend(_collect_conv_layers(layer))
        elif _is_conv_layer(layer):
            layers.append(layer)
    return layers


def find_conv_layer(
    model: tf.keras.Model,
    preferred_name: str | None = None,
) -> tf.keras.layers.Layer:
    """Find a suitable convolutional layer for Grad-CAM.

    Strategy:
        1. If ``preferred_name`` is given and a layer with that name exists
           (including inside nested sub-models), return it.
        2. Otherwise return the last Conv2D layer found by scanning the full
           model depth-first (handles both flat and nested architectures).

    For the Baseline CNN the last Conv2D is "block4_conv".
    For EfficientNetB0 the last Conv2D inside the backbone is typically
    "top_conv".

    Args:
        model: a Keras model (flat or with nested sub-models).
        preferred_name: optional explicit layer name to search for.

    Returns:
        A ``tf.keras.layers.Layer`` suitable as the Grad-CAM target.

    Raises:
        ValueError: if no Conv2D layer can be found.
    """
    all_conv = _collect_conv_layers(model)
    if not all_conv:
        raise ValueError(
            "No Conv2D layers found in the model. "
            "Cannot determine a Grad-CAM target layer."
        )

    if preferred_name is not None:
        # Search flat and nested layers by name.
        for layer in all_conv:
            if layer.name == preferred_name:
                return layer
        # Preferred name not found — fall back to last conv with a warning.
        import warnings
        warnings.warn(
            f"Preferred Grad-CAM layer '{preferred_name}' not found. "
            f"Falling back to last Conv2D: '{all_conv[-1].name}'.",
            stacklevel=2,
        )

    return all_conv[-1]


def _find_sub_model(
    model: tf.keras.Model,
    name_substring: str,
) -> tf.keras.Model | None:
    """Return the first nested sub-model whose name contains name_substring."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and name_substring in layer.name.lower():
            return layer
    return None


# ---------------------------------------------------------------------------
# Grad-CAM model builder
# ---------------------------------------------------------------------------

def _build_grad_model(
    model: tf.keras.Model,
    target_layer: tf.keras.layers.Layer,
) -> tf.keras.Model:
    """Build a Grad-CAM model that outputs [conv_feature_map, predictions].

    This function correctly handles both flat and nested Functional models.

    Flat model (Baseline CNN):
        ``target_layer.output`` is directly in the outer model's graph, so
        the standard one-step ``Model(inputs, [layer.output, model.output])``
        construction works.

    Nested model (EfficientNetB0 transfer):
        ``target_layer`` lives inside a backbone sub-model.
        ``target_layer.output`` is connected to ``backbone.inputs``, NOT to
        the outer model's inputs.  Building
        ``Model(outer_model.inputs, [target_layer.output, outer_model.output])``
        would raise "Graph disconnected" because the two output tensors come
        from different graphs.

        Fix: build an *expanded backbone* sub-model that exposes both the
        conv feature map and the backbone's normal output.  Then rebuild the
        outer model layer-by-layer, substituting the expanded backbone for
        the original one.  Because the backbone internally computes
        backbone_output THROUGH conv_output, the gradient
        ∂class_score/∂conv_output is well-defined and non-None.

    Args:
        model: the outer Keras model (flat or with nested sub-models).
        target_layer: the Conv2D layer to use as the Grad-CAM feature source.

    Returns:
        A Keras model with inputs = model.inputs and
        outputs = [conv_feature_map, predictions].

    Raises:
        ValueError: if target_layer is not reachable in model.
        RuntimeError: if the layer-by-layer rebuild fails to encounter the
            backbone (should not happen for supported architectures).
    """
    # ---- Flat case: target_layer is a direct layer of model ---------------
    # (not itself a Model and directly listed in model.layers)
    direct_non_submodel_layers = [
        l for l in model.layers if not isinstance(l, tf.keras.Model)
    ]
    if any(l is target_layer for l in direct_non_submodel_layers):
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.outputs[0]],
        )

    # ---- Nested case: target_layer is inside a sub-model ------------------
    containing_sub: tf.keras.Model | None = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any(
            l is target_layer for l in layer.layers
        ):
            containing_sub = layer
            break

    if containing_sub is None:
        raise ValueError(
            f"Layer '{target_layer.name}' not found directly in "
            f"model.layers or any immediate nested sub-model.  "
            f"Ensure find_conv_layer() was used to locate the layer."
        )

    # Build expanded sub-model: backbone.inputs → [conv_out, backbone_out].
    # This is valid within the backbone's own internal graph.
    expanded_sub = tf.keras.Model(
        inputs=containing_sub.inputs,
        outputs=[target_layer.output, containing_sub.outputs[0]],
        name="gradcam_expanded_backbone",
    )

    # Rebuild the outer model layer-by-layer, replacing containing_sub with
    # expanded_sub.  For our architectures (linear outer topology) this is
    # straightforward.
    new_input = tf.keras.Input(
        shape=model.input_shape[1:], name="gradcam_outer_input"
    )
    x = new_input
    conv_out: tf.Tensor | None = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer is containing_sub:
            conv_out, x = expanded_sub(x, training=False)
        else:
            # Apply all other layers (Rescaling, GAP, Dropout, Dense, …)
            try:
                x = layer(x, training=False)
            except TypeError:
                # Some layers do not accept training= kwarg.
                x = layer(x)

    if conv_out is None:
        raise RuntimeError(
            "The nested sub-model was not encountered while rebuilding the "
            "outer model.  The outer model topology may have branches that "
            "are not supported by this sequential-rebuild helper."
        )

    return tf.keras.Model(inputs=new_input, outputs=[conv_out, x])


# ---------------------------------------------------------------------------
# Grad-CAM computation
# ---------------------------------------------------------------------------

def compute_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_idx: int | None = None,
    layer: tf.keras.layers.Layer | None = None,
    preferred_layer_name: str | None = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single image.

    Uses tf.GradientTape to compute the gradient of the class score
    (before softmax) with respect to the convolutional feature map, then
    pools gradients spatially and applies ReLU.

    Args:
        model: a compiled Keras model (Baseline CNN or EfficientNetB0 transfer).
            Model weights are NOT modified.
        image: float32 array of shape (H, W, C) or (1, H, W, C).
            Must be in the [0, 1] range that the model expects (the model
            itself applies any internal rescaling adapters).
        class_idx: target class index (0–3).  If None, uses the class with
            the highest predicted probability.
        layer: explicit Conv2D layer to use as the feature source.
            If None, ``find_conv_layer`` is called automatically.
        preferred_layer_name: passed to ``find_conv_layer`` if ``layer``
            is None.

    Returns:
        heatmap: float32 numpy array of shape (H, W) with values in [0, 1].
            Spatial resolution matches the chosen convolutional layer's
            output, upsampled to match the input image H×W.

    Note:
        Model weights are never modified; only forward/backward passes are
        performed through the gradient tape.
    """
    # Ensure (1, H, W, C) batch dimension.
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image_batch = image[np.newaxis, ...]
    else:
        image_batch = image
    image_tensor = tf.constant(image_batch)

    # Resolve target layer.
    if layer is None:
        layer = find_conv_layer(model, preferred_name=preferred_layer_name)

    # Build the Grad-CAM model — handles both flat and nested architectures.
    # For nested models (EfficientNetB0) the standard
    #   Model(outer.inputs, [inner_layer.output, outer.output])
    # raises "Graph disconnected" because inner_layer.output lives in the
    # backbone's own graph.  _build_grad_model resolves this by building an
    # expanded backbone and reconstructing the outer model layer-by-layer.
    grad_model = _build_grad_model(model, layer)

    with tf.GradientTape() as tape:
        # Cast to float32 for tape tracking.
        img_tf = tf.cast(image_tensor, tf.float32)
        tape.watch(img_tf)
        conv_outputs, predictions = grad_model(img_tf, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]).numpy())
        # Use pre-softmax logit if possible, else use softmax output.
        # predictions shape: (1, num_classes)
        class_score = predictions[:, class_idx]

    # Gradients of class score w.r.t. convolutional output.
    grads = tape.gradient(class_score, conv_outputs)  # (1, H', W', C')

    if grads is None:
        raise RuntimeError(
            "Gradient tape returned None. Ensure the chosen layer is part "
            "of the computational graph leading to the model output."
        )

    # Global average pool the gradients over spatial dimensions.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C',)

    # Weight the feature maps and collapse to a 2D heatmap.
    conv_outputs_np = conv_outputs[0].numpy()   # (H', W', C')
    pooled_grads_np = pooled_grads.numpy()       # (C',)

    # heatmap[h, w] = sum_c( weight_c * feature_c[h, w] )
    heatmap = conv_outputs_np @ pooled_grads_np   # (H', W')

    # ReLU: keep only activations that positively influence the class.
    heatmap = np.maximum(heatmap, 0.0)

    # Upsample to input image size.
    h_in = image_batch.shape[1]
    w_in = image_batch.shape[2]
    if heatmap.shape != (h_in, w_in):
        heatmap_tensor = tf.image.resize(
            heatmap[..., np.newaxis],
            (h_in, w_in),
            method="bilinear",
        ).numpy()[..., 0]
    else:
        heatmap_tensor = heatmap

    # Normalise to [0, 1].  Handle the degenerate all-zero case safely.
    max_val = heatmap_tensor.max()
    if max_val > 1e-8:
        heatmap_tensor = heatmap_tensor / max_val
    else:
        heatmap_tensor = np.zeros_like(heatmap_tensor)

    return heatmap_tensor.astype(np.float32)


# ---------------------------------------------------------------------------
# Overlay utility
# ---------------------------------------------------------------------------

def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a Grad-CAM heatmap with the original MRI image.

    Args:
        image: float32 array (H, W, C) in [0, 1].  Channels are preserved
               (RGB or grayscale).
        heatmap: float32 array (H, W) in [0, 1] from ``compute_gradcam``.
        alpha: blending weight for the heatmap overlay (0=original only,
               1=heatmap only).  Default 0.4.
        colormap: matplotlib colormap name applied to the heatmap.
               Default "jet".

    Returns:
        float32 array (H, W, 3) in [0, 1] — the blended RGB image.

    Note:
        This function requires ``matplotlib`` for the colormap lookup.
    """
    import matplotlib as mpl
    if hasattr(mpl, "colormaps"):
        cmap = mpl.colormaps[colormap]
    else:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
    heatmap_rgb = cmap(heatmap)[..., :3].astype(np.float32)  # (H, W, 3)

    # Ensure image is RGB (H, W, 3).
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    # Alpha blend.
    blended = (1.0 - alpha) * image + alpha * heatmap_rgb
    return np.clip(blended, 0.0, 1.0).astype(np.float32)
