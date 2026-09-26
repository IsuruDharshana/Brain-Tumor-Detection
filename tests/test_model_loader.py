"""
test_model_loader.py
--------------------
Phase 7 unit tests for model path resolution, atomic downloading,
SHA-256 integrity verification, and Keras model loader.

Requirements verified:
  - Existing local model path is used directly without download
  - BRAIN_TUMOR_MODEL_PATH override wins when present
  - Missing local model + BRAIN_TUMOR_MODEL_URL triggers download
  - Existing model is not redownloaded
  - Missing URL with no local model raises clear FileNotFoundError
  - Temporary download file (.tmp) is atomically renamed to final model
  - Failed download leaves no partial final model or leftover temp file
  - Optional SHA-256 verification succeeds on match
  - Optional SHA-256 mismatch raises error and removes temp file
  - Model loader calls tf.keras.models.load_model(..., compile=False)

NO internet access or downloads.
NO real .keras model required (tests use mock downloads and synthetic files).
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from src.app.model_loader import (
    DEFAULT_MODEL_REL_PATH,
    ENV_MODEL_PATH_VAR,
    ENV_MODEL_SHA256_VAR,
    ENV_MODEL_URL_VAR,
    compute_file_sha256,
    download_model_file,
    get_model_path,
    load_prediction_model,
    resolve_model_path,
    verify_file_sha256,
)


class MockUrlOpenResponse:
    """Mock context manager for urllib.request.urlopen."""

    def __init__(self, data: bytes, chunk_size: int = 4096, fail_midway: bool = False):
        self.data = data
        self.chunk_size = chunk_size
        self.fail_midway = fail_midway
        self.stream = io.BytesIO(data)
        self.bytes_read = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def read(self, size: int = -1) -> bytes:
        if self.fail_midway and self.bytes_read > 0:
            raise urllib.error.URLError("Simulated mid-stream connection drop")
        chunk = self.stream.read(min(size, self.chunk_size) if size > 0 else self.chunk_size)
        self.bytes_read += len(chunk)
        return chunk


# ---------------------------------------------------------------------------
# Path Resolution Tests
# ---------------------------------------------------------------------------

def test_existing_local_model_path_is_used(tmp_path, monkeypatch):
    """When the default local model exists and no env vars are set, use it directly."""
    fake_default = tmp_path / "models" / "baseline_cnn.keras"
    fake_default.parent.mkdir(parents=True, exist_ok=True)
    fake_default.write_bytes(b"local_model_content")

    monkeypatch.setattr("src.app.model_loader.DEFAULT_MODEL_PATH", fake_default)
    monkeypatch.delenv(ENV_MODEL_PATH_VAR, raising=False)
    monkeypatch.delenv(ENV_MODEL_URL_VAR, raising=False)
    monkeypatch.delenv(ENV_MODEL_SHA256_VAR, raising=False)

    resolved = resolve_model_path()
    assert resolved == fake_default
    assert resolved.is_file()


def test_brain_tumor_model_path_override_wins(tmp_path, monkeypatch):
    """BRAIN_TUMOR_MODEL_PATH pointing to an existing file overrides default path."""
    fake_default = tmp_path / "default_model.keras"
    fake_default.write_bytes(b"default")

    fake_custom = tmp_path / "custom_dir" / "my_custom_model.keras"
    fake_custom.parent.mkdir(parents=True, exist_ok=True)
    fake_custom.write_bytes(b"custom_model")

    monkeypatch.setattr("src.app.model_loader.DEFAULT_MODEL_PATH", fake_default)
    monkeypatch.setenv(ENV_MODEL_PATH_VAR, str(fake_custom))
    monkeypatch.delenv(ENV_MODEL_URL_VAR, raising=False)

    resolved = resolve_model_path()
    assert resolved == fake_custom
    assert resolved.read_bytes() == b"custom_model"


def test_missing_local_model_plus_url_triggers_download(tmp_path, monkeypatch):
    """When local model is missing and BRAIN_TUMOR_MODEL_URL is set, download is called."""
    missing_default = tmp_path / "models" / "baseline_cnn.keras"
    monkeypatch.setattr("src.app.model_loader.DEFAULT_MODEL_PATH", missing_default)
    monkeypatch.delenv(ENV_MODEL_PATH_VAR, raising=False)
    monkeypatch.setenv(ENV_MODEL_URL_VAR, "https://mock.example.com/assets/baseline_cnn.keras")

    with patch("src.app.model_loader.download_model_file") as mock_download:
        mock_download.return_value = missing_default
        resolved = resolve_model_path()

        mock_download.assert_called_once_with(
            url="https://mock.example.com/assets/baseline_cnn.keras",
            destination=missing_default,
            expected_sha256=None,
        )
        assert resolved == missing_default


def test_existing_model_does_not_redownload(tmp_path):
    """download_model_file returns existing file immediately without network request."""
    dest = tmp_path / "baseline_cnn.keras"
    dest.write_bytes(b"existing_file_content")

    with patch("urllib.request.urlopen") as mock_urlopen:
        resolved = download_model_file(
            url="https://mock.example.com/model.keras",
            destination=dest,
        )
        mock_urlopen.assert_not_called()
        assert resolved == dest


def test_missing_url_with_no_model_raises_clear_error(tmp_path, monkeypatch):
    """When neither local model nor URL is available, raise FileNotFoundError."""
    missing_path = tmp_path / "models" / "nonexistent.keras"
    monkeypatch.setattr("src.app.model_loader.DEFAULT_MODEL_PATH", missing_path)
    monkeypatch.delenv(ENV_MODEL_PATH_VAR, raising=False)
    monkeypatch.delenv(ENV_MODEL_URL_VAR, raising=False)

    with pytest.raises(FileNotFoundError, match="Trained model checkpoint not found"):
        resolve_model_path()


# ---------------------------------------------------------------------------
# Atomic Download & Error Isolation Tests
# ---------------------------------------------------------------------------

def test_temporary_download_is_atomically_renamed(tmp_path):
    """Download writes to a .tmp file first and renames to destination upon success."""
    dest = tmp_path / "baseline_cnn.keras"
    content = b"Keras_model_binary_payload_12345"

    with patch("urllib.request.urlopen", return_value=MockUrlOpenResponse(content)):
        resolved = download_model_file(
            url="https://mock.example.com/model.keras",
            destination=dest,
        )

        assert resolved == dest
        assert dest.is_file()
        assert dest.read_bytes() == content
        # Temporary file must not remain
        tmp_file = dest.with_name(f"{dest.name}.tmp")
        assert not tmp_file.exists()


def test_failed_download_leaves_no_partial_final_model(tmp_path):
    """Network failure leaves no destination file and cleans up temporary file."""
    dest = tmp_path / "baseline_cnn.keras"
    partial_content = b"partial_header_data"

    with patch("urllib.request.urlopen", return_value=MockUrlOpenResponse(partial_content, fail_midway=True)):
        with pytest.raises(RuntimeError, match="Model download failed"):
            download_model_file(
                url="https://mock.example.com/model.keras",
                destination=dest,
            )

        # Destination must never exist after a failed download
        assert not dest.exists()
        # Temporary file must be cleaned up
        tmp_file = dest.with_name(f"{dest.name}.tmp")
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# SHA-256 Integrity Verification Tests
# ---------------------------------------------------------------------------

def test_optional_sha256_success(tmp_path):
    """Download succeeds when computed SHA-256 matches expected hash."""
    dest = tmp_path / "baseline_cnn.keras"
    content = b"certified_valid_model_data"
    expected_hash = hashlib.sha256(content).hexdigest()

    with patch("urllib.request.urlopen", return_value=MockUrlOpenResponse(content)):
        resolved = download_model_file(
            url="https://mock.example.com/model.keras",
            destination=dest,
            expected_sha256=expected_hash.upper(),  # test case-insensitivity
        )

        assert resolved.is_file()
        assert dest.read_bytes() == content


def test_optional_sha256_mismatch_raises_error(tmp_path):
    """Hash mismatch raises error, deletes temp file, and does not create final file."""
    dest = tmp_path / "baseline_cnn.keras"
    content = b"tampered_or_corrupt_data"
    wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"

    with patch("urllib.request.urlopen", return_value=MockUrlOpenResponse(content)):
        with pytest.raises(RuntimeError, match="integrity check"):
            download_model_file(
                url="https://mock.example.com/model.keras",
                destination=dest,
                expected_sha256=wrong_hash,
            )

        assert not dest.exists()
        tmp_file = dest.with_name(f"{dest.name}.tmp")
        assert not tmp_file.exists()


def test_existing_file_sha256_mismatch_rejected(tmp_path, monkeypatch):
    """Existing file with mismatched hash is rejected with ValueError."""
    fake_model = tmp_path / "corrupt_local.keras"
    fake_model.write_bytes(b"corrupt_bits")

    monkeypatch.setenv(ENV_MODEL_PATH_VAR, str(fake_model))
    monkeypatch.setenv(ENV_MODEL_SHA256_VAR, "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890")

    with pytest.raises(ValueError, match="failed SHA-256 integrity verification"):
        resolve_model_path()


# ---------------------------------------------------------------------------
# Keras Loader Invocation Test
# ---------------------------------------------------------------------------

def test_load_prediction_model_calls_keras_compile_false(tmp_path, monkeypatch):
    """load_prediction_model must invoke tf.keras.models.load_model with compile=False."""
    fake_model = tmp_path / "baseline_cnn.keras"
    fake_model.write_bytes(b"mock_model")

    monkeypatch.setattr("src.app.model_loader.DEFAULT_MODEL_PATH", fake_model)
    monkeypatch.delenv(ENV_MODEL_PATH_VAR, raising=False)
    monkeypatch.delenv(ENV_MODEL_URL_VAR, raising=False)
    monkeypatch.delenv(ENV_MODEL_SHA256_VAR, raising=False)

    mock_keras_model = MagicMock()
    with patch("tensorflow.keras.models.load_model", return_value=mock_keras_model) as mock_load:
        loaded = load_prediction_model(fake_model)

        mock_load.assert_called_once_with(str(fake_model), compile=False)
        assert loaded == mock_keras_model
