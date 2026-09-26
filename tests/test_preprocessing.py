"""
test_preprocessing.py
---------------------
Unit tests for the Phase 3 base preprocessing (src/data/preprocessing.py):
decode -> RGB conversion -> 224x224 resize -> float32 [0, 1] normalisation.

All images are synthetic (created on the fly), so the full Kaggle dataset is
not needed.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
from PIL import Image
import tensorflow as tf

from src.config import CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
from src.data.preprocessing import (
    decode_and_preprocess_image_bytes,
    load_and_preprocess_image,
)

EXPECTED_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)


def _save(path: Path, mode: str, color, size=(48, 32), image_format=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path, format=image_format)
    return path


def test_grayscale_jpg_becomes_three_channel_rgb(tmp_path: Path) -> None:
    """L-mode JPEG -> RGB; the three channels must be replicas of each other."""
    path = _save(tmp_path / "gray.jpg", "L", 128)
    image = load_and_preprocess_image(str(path))

    assert image.shape == EXPECTED_SHAPE
    assert image.dtype == np.float32
    np.testing.assert_allclose(image.numpy()[..., 0], image.numpy()[..., 1])
    np.testing.assert_allclose(image.numpy()[..., 1], image.numpy()[..., 2])


def test_rgba_png_is_converted_to_rgb(tmp_path: Path) -> None:
    """RGBA PNG (alpha dropped) -> 3 channels."""
    path = _save(tmp_path / "rgba.png", "RGBA", (10, 20, 30, 40))
    image = load_and_preprocess_image(str(path))
    assert image.shape == EXPECTED_SHAPE
    assert image.dtype == np.float32


def test_palette_png_is_converted_to_rgb(tmp_path: Path) -> None:
    """Palette (P) PNG -> 3 channels."""
    path = tmp_path / "palette.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), (200, 100, 50)).convert("P").save(path)
    image = load_and_preprocess_image(str(path))
    assert image.shape == EXPECTED_SHAPE


def test_png_content_with_jpg_extension_decodes(tmp_path: Path) -> None:
    """The real dataset contains PNG-content files with a .jpg extension."""
    path = tmp_path / "misleading.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), (90, 120, 30, 255)).save(path, format="PNG")

    assert Image.open(path).format == "PNG"  # confirms the setup
    image = load_and_preprocess_image(str(path))
    assert image.shape == EXPECTED_SHAPE
    assert image.dtype == np.float32


@pytest.mark.parametrize("size", [(37, 91), (300, 150), (224, 224), (150, 167)])
def test_resize_to_224x224(tmp_path: Path, size: tuple[int, int]) -> None:
    """Different source dimensions all end up 224x224."""
    path = _save(tmp_path / f"resize_{size[0]}x{size[1]}.png", "RGB", (1, 2, 3), size=size)
    image = load_and_preprocess_image(str(path))
    assert image.shape == EXPECTED_SHAPE


@pytest.mark.parametrize("value,expected", [(0, 0.0), (128, 128 / 255), (255, 1.0)])
def test_normalization_divides_by_255(tmp_path: Path, value: int, expected: float) -> None:
    """A solid-colour image keeps its value after resize, scaled to [0, 1]."""
    path = _save(tmp_path / f"solid_{value}.png", "L", value)
    image = load_and_preprocess_image(str(path)).numpy()

    assert image.min() >= 0.0 and image.max() <= 1.0
    np.testing.assert_allclose(image, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Byte Preprocessing & Equivalence Tests
# ---------------------------------------------------------------------------

def test_decode_and_preprocess_image_bytes_png() -> None:
    """PNG bytes decode to (224, 224, 3) float32 in [0, 1]."""
    buf = io.BytesIO()
    Image.new("RGB", (60, 80), color=(100, 150, 200)).save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    image = decode_and_preprocess_image_bytes(raw_bytes)
    assert image.shape == EXPECTED_SHAPE
    assert image.dtype == np.float32
    assert float(tf.reduce_min(image)) >= 0.0
    assert float(tf.reduce_max(image)) <= 1.0


def test_decode_and_preprocess_image_bytes_jpeg() -> None:
    """JPEG bytes decode to (224, 224, 3) float32 in [0, 1]."""
    buf = io.BytesIO()
    Image.new("RGB", (80, 60), color=(50, 75, 100)).save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    image = decode_and_preprocess_image_bytes(raw_bytes)
    assert image.shape == EXPECTED_SHAPE
    assert image.dtype == np.float32


def test_decode_and_preprocess_image_bytes_grayscale_becomes_rgb() -> None:
    """Grayscale 'L' bytes decode to 3 identical channels."""
    buf = io.BytesIO()
    Image.new("L", (50, 50), color=120).save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    image = decode_and_preprocess_image_bytes(raw_bytes).numpy()
    assert image.shape == EXPECTED_SHAPE
    np.testing.assert_allclose(image[..., 0], image[..., 1])
    np.testing.assert_allclose(image[..., 1], image[..., 2])


def test_decode_and_preprocess_image_bytes_rgba_becomes_rgb() -> None:
    """RGBA bytes decode to 3-channel RGB without alpha."""
    buf = io.BytesIO()
    Image.new("RGBA", (64, 64), color=(10, 20, 30, 200)).save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    image = decode_and_preprocess_image_bytes(raw_bytes)
    assert image.shape == EXPECTED_SHAPE
    assert image.dtype == np.float32


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_byte_vs_path_exact_equivalence(tmp_path: Path, fmt: str) -> None:
    """load_and_preprocess_image and decode_and_preprocess_image_bytes produce identical tensors."""
    img = Image.new("RGB", (95, 120), color=(45, 90, 135))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    raw_bytes = buf.getvalue()

    file_path = tmp_path / f"equiv_test.{fmt.lower()}"
    file_path.write_bytes(raw_bytes)

    from_path = load_and_preprocess_image(str(file_path)).numpy()
    from_bytes = decode_and_preprocess_image_bytes(raw_bytes).numpy()

    np.testing.assert_allclose(from_bytes, from_path, atol=1e-6)
