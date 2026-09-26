"""
test_app_inference.py
---------------------
Phase 7 unit tests for Streamlit application preprocessing and inference logic.

Requirements verified:
  - Basic MRI plausibility validation rejects strongly colourful and blank images
  - Grayscale and mildly tinted images pass the conservative validation gate
  - Preprocessing produces (1, 224, 224, 3) float32 in [0, 1]
  - Grayscale ('L'), RGBA, and Palette ('P') convert to 3-channel RGB
  - Black image maps to 0.0, White image maps to 1.0
  - Prediction class mapping is strictly preserved (0=glioma, 1=meningioma, 2=notumor, 3=pituitary)
  - Confidence equals max probability
  - Exactly 4 class probabilities returned
  - Invalid output shape, NaN/Inf probabilities, and out-of-range values raise ValueError
  - Model path resolution handles defaults and BRAIN_TUMOR_MODEL_PATH override
  - Missing model file raises clean FileNotFoundError

NO real MRI images or test_manifest.csv used.
NO real .keras model required (tests use mock callables).
NO internet access or downloads.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.app.inference import (
    CLASS_DISPLAY_NAMES,
    predict_image,
    preprocess_image,
    validate_mri_like_image,
)
from src.data.preprocessing import (
    decode_and_preprocess_image_bytes,
    load_and_preprocess_image,
)
from src.app.model_loader import (
    DEFAULT_MODEL_REL_PATH,
    ENV_MODEL_PATH_VAR,
    get_model_path,
    load_prediction_model,
)
from src.config import CLASS_NAMES


# ---------------------------------------------------------------------------
# Input Plausibility Validation Tests
# ---------------------------------------------------------------------------

def _grayscale_gradient(size: int = 64) -> np.ndarray:
    """Create a non-uniform synthetic grayscale image without dataset files."""
    row = np.linspace(10, 245, size, dtype=np.uint8)
    return np.tile(row, (size, 1))


def test_validate_grayscale_synthetic_image_passes():
    image = Image.fromarray(_grayscale_gradient())

    result = validate_mri_like_image(image)

    assert result["is_plausible_mri"] is True


def test_validate_equal_rgb_channels_passes():
    grayscale = _grayscale_gradient()
    equal_rgb = np.repeat(grayscale[:, :, np.newaxis], 3, axis=2)
    image = Image.fromarray(equal_rgb, mode="RGB")

    result = validate_mri_like_image(image)

    assert result["is_plausible_mri"] is True
    assert result["metrics"]["mean_channel_difference"] == pytest.approx(0.0)


@pytest.mark.parametrize("color", [(255, 0, 0), (0, 0, 255)])
def test_validate_strong_single_color_fails(color):
    image = Image.new("RGB", (64, 64), color=color)

    result = validate_mri_like_image(image)

    assert result["is_plausible_mri"] is False
    assert "colour" in result["reason"].lower()


def test_validate_highly_saturated_multicolor_image_fails():
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, 0:16] = (255, 0, 0)
    pixels[:, 16:32] = (0, 255, 0)
    pixels[:, 32:48] = (0, 0, 255)
    pixels[:, 48:64] = (255, 255, 0)
    image = Image.fromarray(pixels, mode="RGB")

    result = validate_mri_like_image(image)

    assert result["is_plausible_mri"] is False
    assert result["metrics"]["colorful_pixel_fraction"] == pytest.approx(1.0)


def test_validate_nearly_grayscale_slightly_tinted_image_passes():
    grayscale = _grayscale_gradient().astype(np.int16)
    tinted = np.stack(
        [
            np.clip(grayscale + 4, 0, 255),
            grayscale,
            np.clip(grayscale - 3, 0, 255),
        ],
        axis=2,
    ).astype(np.uint8)
    image = Image.fromarray(tinted, mode="RGB")

    result = validate_mri_like_image(image)

    assert result["is_plausible_mri"] is True


@pytest.mark.parametrize("value", [0, 255])
def test_validate_blank_black_or_white_image_fails(value):
    image = Image.new("RGB", (64, 64), color=(value, value, value))

    result = validate_mri_like_image(image)

    assert result["is_plausible_mri"] is False
    assert "blank" in result["reason"].lower()


def test_validate_result_has_expected_schema():
    result = validate_mri_like_image(Image.fromarray(_grayscale_gradient()))

    assert set(result) == {"is_plausible_mri", "reason", "metrics"}
    assert isinstance(result["is_plausible_mri"], bool)
    assert isinstance(result["reason"], str)
    assert set(result["metrics"]) == {
        "mean_saturation",
        "p95_saturation",
        "mean_channel_difference",
        "colorful_pixel_fraction",
        "intensity_std",
        "intensity_range",
    }
    assert all(isinstance(value, float) for value in result["metrics"].values())


def test_failed_validation_guard_does_not_call_model():
    class FailIfCalledModel:
        def __init__(self):
            self.call_count = 0

        def __call__(self, x, training=False):
            self.call_count += 1
            raise AssertionError("Rejected input reached model prediction")

    image = Image.new("RGB", (64, 64), color=(255, 0, 0))
    model = FailIfCalledModel()
    validation = validate_mri_like_image(image)

    # This is the same rejection guard used by app.py before preprocessing.
    if validation["is_plausible_mri"]:
        predict_image(model, preprocess_image(image))

    assert validation["is_plausible_mri"] is False
    assert model.call_count == 0


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

def test_preprocess_rgb_image_shape_and_dtype():
    """RGB image must produce shape (1, 224, 224, 3) and float32 dtype."""
    img = Image.new("RGB", (100, 150), color=(120, 130, 140))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_preprocess_grayscale_converts_to_three_channels():
    """Grayscale 'L' mode must convert to 3-channel RGB."""
    img = Image.new("L", (80, 80), color=128)
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 224, 224, 3)
    # In RGB conversion of gray=128, all 3 channels should be identical
    np.testing.assert_allclose(tensor[0, :, :, 0], tensor[0, :, :, 1])
    np.testing.assert_allclose(tensor[0, :, :, 1], tensor[0, :, :, 2])


def test_preprocess_rgba_converts_to_rgb():
    """RGBA mode must convert to 3-channel RGB without alpha."""
    img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 224, 224, 3)


def test_preprocess_palette_converts_to_rgb():
    """Palette 'P' mode must convert to 3-channel RGB."""
    img = Image.new("P", (64, 64))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 224, 224, 3)


def test_preprocess_output_range():
    """Pixel values must be strictly normalized into [0.0, 1.0]."""
    img = Image.new("RGB", (50, 50), color=(50, 100, 200))
    tensor = preprocess_image(img)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocess_black_image_maps_to_zero():
    """All-black image (0, 0, 0) must normalize to exactly 0.0."""
    img = Image.new("RGB", (32, 32), color=(0, 0, 0))
    tensor = preprocess_image(img)
    np.testing.assert_allclose(tensor, 0.0, atol=1e-6)


def test_preprocess_white_image_maps_to_one():
    """All-white image (255, 255, 255) must normalize to exactly 1.0."""
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    tensor = preprocess_image(img)
    np.testing.assert_allclose(tensor, 1.0, atol=1e-6)


def test_preprocess_bytes_input():
    """Preprocess must accept raw bytes encoded in PNG or JPEG."""
    img = Image.new("RGB", (40, 40), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    tensor = preprocess_image(raw_bytes)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_preprocess_bytesio_stream():
    """Preprocess must accept BytesIO streams."""
    img = Image.new("RGB", (40, 40), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    tensor = preprocess_image(buf)
    assert tensor.shape == (1, 224, 224, 3)


def test_preprocess_invalid_data_raises_value_error():
    """Corrupt or non-image data must raise ValueError."""
    corrupt_bytes = b"not_a_valid_image_file_content"
    with pytest.raises(ValueError, match="Could not decode image"):
        preprocess_image(corrupt_bytes)


def test_preprocess_bytearray_input():
    """Preprocess must accept bytearray input."""
    img = Image.new("RGB", (40, 40), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = bytearray(buf.getvalue())

    tensor = preprocess_image(raw_bytes)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_preprocess_filepath_input(tmp_path: Path):
    """Preprocess must accept file paths as str or Path."""
    img = Image.new("RGB", (50, 50), color=(30, 60, 90))
    file_path = tmp_path / "test_file.png"
    img.save(file_path, format="PNG")

    tensor_str = preprocess_image(str(file_path))
    tensor_path = preprocess_image(file_path)

    assert tensor_str.shape == (1, 224, 224, 3)
    assert tensor_path.shape == (1, 224, 224, 3)
    np.testing.assert_allclose(tensor_str, tensor_path, atol=1e-6)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_preprocess_app_matches_phase3_exact(tmp_path: Path, fmt: str):
    """App preprocessing of raw bytes matches Phase 3 preprocessing exactly."""
    img = Image.new("RGB", (77, 99), color=(40, 80, 120))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    raw_bytes = buf.getvalue()

    file_path = tmp_path / f"test_match.{fmt.lower()}"
    file_path.write_bytes(raw_bytes)

    phase3_out = load_and_preprocess_image(str(file_path)).numpy()
    app_bytes_out = preprocess_image(raw_bytes)[0]
    app_path_out = preprocess_image(file_path)[0]

    np.testing.assert_allclose(app_bytes_out, phase3_out, atol=1e-6)
    np.testing.assert_allclose(app_path_out, phase3_out, atol=1e-6)


def test_preprocess_empty_bytes_raises_value_error():
    """Empty bytes or bytearray must raise ValueError."""
    with pytest.raises(ValueError, match="Empty image bytes provided"):
        preprocess_image(b"")


# ---------------------------------------------------------------------------
# Prediction Inference Tests (Using Mock Models)
# ---------------------------------------------------------------------------

class MockPredictModel:
    """Mock model returning pre-defined softmax probability arrays."""

    def __init__(self, probabilities: list[float] | np.ndarray):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        return self.probabilities


def test_predict_class_mapping_glioma():
    """Top probability at index 0 must map to glioma."""
    model = MockPredictModel([[0.80, 0.10, 0.05, 0.05]])
    inp = np.zeros((1, 224, 224, 3), dtype=np.float32)
    res = predict_image(model, inp)

    assert res["predicted_class_id"] == 0
    assert res["predicted_class_name"] == "glioma"
    assert res["predicted_display_name"] == "Glioma"
    assert res["confidence"] == pytest.approx(0.80)


def test_predict_class_mapping_pituitary():
    """Top probability at index 3 must map to pituitary."""
    model = MockPredictModel([[0.05, 0.10, 0.15, 0.70]])
    inp = np.zeros((1, 224, 224, 3), dtype=np.float32)
    res = predict_image(model, inp)

    assert res["predicted_class_id"] == 3
    assert res["predicted_class_name"] == "pituitary"
    assert res["predicted_display_name"] == "Pituitary"
    assert res["confidence"] == pytest.approx(0.70)


def test_predict_confidence_is_max_prob():
    """Confidence must strictly match the maximum class probability."""
    model = MockPredictModel([[0.15, 0.65, 0.10, 0.10]])
    inp = np.zeros((1, 224, 224, 3), dtype=np.float32)
    res = predict_image(model, inp)

    assert res["confidence"] == pytest.approx(0.65)
    assert res["predicted_class_id"] == 1


def test_predict_four_class_probabilities_returned():
    """All 4 classes must be returned in canonical and display probability dicts."""
    model = MockPredictModel([[0.25, 0.25, 0.25, 0.25]])
    inp = np.zeros((1, 224, 224, 3), dtype=np.float32)
    res = predict_image(model, inp)

    assert set(res["probabilities"].keys()) == set(CLASS_NAMES)
    assert set(res["display_probabilities"].keys()) == set(CLASS_DISPLAY_NAMES.values())
    assert len(res["probabilities"]) == 4


def test_predict_invalid_output_shape_raises():
    """Model output with != 4 classes must raise ValueError."""
    model = MockPredictModel([[0.5, 0.5]])  # 2 classes instead of 4
    inp = np.zeros((1, 224, 224, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected model output of shape"):
        predict_image(model, inp)


def test_predict_nan_probability_rejected():
    """Model output containing NaN must raise ValueError."""
    model = MockPredictModel([[0.5, np.nan, 0.2, 0.3]])
    inp = np.zeros((1, 224, 224, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or infinite"):
        predict_image(model, inp)


def test_predict_inf_probability_rejected():
    """Model output containing Inf must raise ValueError."""
    model = MockPredictModel([[0.5, np.inf, 0.2, 0.3]])
    inp = np.zeros((1, 224, 224, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or infinite"):
        predict_image(model, inp)


def test_predict_out_of_range_probability_rejected():
    """Model output outside [0, 1] must raise ValueError."""
    model = MockPredictModel([[1.5, -0.2, 0.1, 0.1]])
    inp = np.zeros((1, 224, 224, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="outside valid probability range"):
        predict_image(model, inp)


# ---------------------------------------------------------------------------
# Model Loader Tests
# ---------------------------------------------------------------------------

def test_get_model_path_default():
    """Without env override, get_model_path must point to models/baseline_cnn.keras."""
    # Ensure env variable is unset for this test
    old_val = os.environ.pop(ENV_MODEL_PATH_VAR, None)
    try:
        path = get_model_path()
        assert path.name == "baseline_cnn.keras"
        assert "models" in path.parts
    finally:
        if old_val is not None:
            os.environ[ENV_MODEL_PATH_VAR] = old_val


def test_get_model_path_env_override(tmp_path):
    """Setting BRAIN_TUMOR_MODEL_PATH must override the default path."""
    fake_model_file = tmp_path / "custom_model.keras"
    os.environ[ENV_MODEL_PATH_VAR] = str(fake_model_file)
    try:
        path = get_model_path()
        assert path == fake_model_file
    finally:
        os.environ.pop(ENV_MODEL_PATH_VAR, None)


def test_load_prediction_model_missing_file_raises():
    """Attempting to load a non-existent model file must raise FileNotFoundError."""
    missing_path = Path("/nonexistent/directory/fake_model.keras")
    with pytest.raises(FileNotFoundError, match="Trained model checkpoint not found"):
        load_prediction_model(model_path=missing_path)
