"""
model_loader.py
---------------
Model loading and acquisition utilities for the Phase 7 Streamlit application.

Supports:
  - Local resolution priority:
      1. BRAIN_TUMOR_MODEL_PATH if set and file exists
      2. Default local path: models/baseline_cnn.keras (if it exists)
      3. If local model missing and BRAIN_TUMOR_MODEL_URL is set: download model
      4. Otherwise raise a clear FileNotFoundError
  - Deployment-safe atomic download via urllib.request (standard library)
  - Temporary file download (.tmp) with atomic replacement to prevent partial files
  - Optional integrity verification via BRAIN_TUMOR_MODEL_SHA256
  - Non-destructive uncompiled loading: tf.keras.models.load_model(..., compile=False)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
import urllib.request

from src.config import REPO_ROOT

ENV_MODEL_PATH_VAR = "BRAIN_TUMOR_MODEL_PATH"
ENV_MODEL_URL_VAR = "BRAIN_TUMOR_MODEL_URL"
ENV_MODEL_SHA256_VAR = "BRAIN_TUMOR_MODEL_SHA256"

DEFAULT_MODEL_REL_PATH = Path("models") / "baseline_cnn.keras"
DEFAULT_MODEL_PATH = REPO_ROOT / DEFAULT_MODEL_REL_PATH
DEFAULT_DOWNLOAD_TIMEOUT = 60.0


def compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hexadecimal digest of a file.

    Args:
        file_path: path to the file to hash.

    Returns:
        Lowercase hexadecimal string of the SHA-256 hash.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest().lower()


def verify_file_sha256(file_path: Path, expected_sha256: str) -> bool:
    """Verify that a file matches an expected SHA-256 hash.

    Args:
        file_path: path to the file to verify.
        expected_sha256: expected hexadecimal SHA-256 string (case-insensitive).

    Returns:
        True if hashes match, False otherwise.
    """
    if not expected_sha256 or not expected_sha256.strip():
        return True
    actual = compute_file_sha256(file_path)
    return actual == expected_sha256.strip().lower()


def download_model_file(
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
) -> Path:
    """Atomically download a model file from a URL to destination path.

    Downloads to a temporary file first and renames to the destination path
    only after successful completion and optional SHA-256 verification.
    If the destination file already exists, it is not redownloaded.

    Args:
        url: direct HTTP/HTTPS URL pointing to the model asset.
        destination: target Path where the model should reside.
        expected_sha256: optional expected SHA-256 hash.
        timeout: network timeout in seconds (default 60.0).

    Returns:
        Path to the verified model file at destination.

    Raises:
        ValueError: if expected_sha256 is provided and verification fails.
        RuntimeError: if the network download fails.
    """
    # If destination already exists, verify hash if required, and return
    if destination.is_file():
        if expected_sha256:
            if not verify_file_sha256(destination, expected_sha256):
                raise ValueError(
                    f"Existing model file at '{destination}' failed SHA-256 integrity verification."
                )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f"{destination.name}.tmp")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "BrainTumorDetection-StreamlitApp/1.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, open(tmp_path, "wb") as out_file:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                out_file.write(chunk)

        # Integrity verification prior to renaming
        if expected_sha256:
            if not verify_file_sha256(tmp_path, expected_sha256):
                actual_hash = compute_file_sha256(tmp_path)
                raise ValueError(
                    f"Downloaded model failed SHA-256 integrity check. "
                    f"Expected {expected_sha256.lower()}, got {actual_hash}."
                )

        # Atomic replacement: rename temporary file to destination
        os.replace(tmp_path, destination)
        return destination

    except Exception as exc:
        raise RuntimeError(f"Model download failed from configured URL: {exc}") from exc

    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def get_model_path() -> Path:
    """Resolve the default or configured path to the trained model file.

    Checks the BRAIN_TUMOR_MODEL_PATH environment variable first.
    If unset or empty, defaults to models/baseline_cnn.keras relative to REPO_ROOT.

    Returns:
        Path object pointing to the expected model file location.
    """
    env_path = os.environ.get(ENV_MODEL_PATH_VAR, "").strip()
    if env_path:
        return Path(env_path)
    return DEFAULT_MODEL_PATH


def resolve_model_path(model_path: Path | str | None = None) -> Path:
    """Resolve the model file location following deployment priority rules.

    Priority:
      1. BRAIN_TUMOR_MODEL_PATH if set and file exists
      2. Default local path (models/baseline_cnn.keras) if it exists
      3. If local model missing and BRAIN_TUMOR_MODEL_URL is set: download model
      4. Otherwise raise FileNotFoundError with a clear explanation

    Args:
        model_path: optional explicit path. If provided and file exists,
            it takes precedence.

    Returns:
        Path to an existing, verified model file on disk.

    Raises:
        FileNotFoundError: if no local model exists and BRAIN_TUMOR_MODEL_URL is unset.
        ValueError: if BRAIN_TUMOR_MODEL_SHA256 is set and hash check fails.
        RuntimeError: if download fails.
    """
    expected_sha256 = os.environ.get(ENV_MODEL_SHA256_VAR, "").strip() or None
    model_url = os.environ.get(ENV_MODEL_URL_VAR, "").strip()

    # 1. Explicit argument if provided
    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            if expected_sha256 and not verify_file_sha256(p, expected_sha256):
                raise ValueError(
                    f"Model file at '{p}' failed SHA-256 integrity verification."
                )
            return p
        if model_url:
            return download_model_file(
                url=model_url,
                destination=p,
                expected_sha256=expected_sha256,
            )
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {p}\n"
            f"Please verify that the model checkpoint exists, or configure '{ENV_MODEL_URL_VAR}'."
        )

    # 2. BRAIN_TUMOR_MODEL_PATH if set and file exists
    env_path_str = os.environ.get(ENV_MODEL_PATH_VAR, "").strip()
    if env_path_str:
        env_path = Path(env_path_str)
        if env_path.is_file():
            if expected_sha256 and not verify_file_sha256(env_path, expected_sha256):
                raise ValueError(
                    f"Model file at '{env_path}' failed SHA-256 integrity verification."
                )
            return env_path

    # 3. Default local path if it exists
    default_path = DEFAULT_MODEL_PATH
    if default_path.is_file():
        if expected_sha256 and not verify_file_sha256(default_path, expected_sha256):
            raise ValueError(
                f"Model file at '{default_path}' failed SHA-256 integrity verification."
            )
        return default_path

    # 4. If local model missing and BRAIN_TUMOR_MODEL_URL is set: download model
    if model_url:
        target_dest = Path(env_path_str) if env_path_str else default_path
        return download_model_file(
            url=model_url,
            destination=target_dest,
            expected_sha256=expected_sha256,
        )

    # 5. Otherwise raise a clear error
    target_desc = env_path_str if env_path_str else str(default_path)
    raise FileNotFoundError(
        f"Trained model checkpoint not found at: {target_desc}\n"
        f"Local model file is missing and '{ENV_MODEL_URL_VAR}' is not configured "
        f"for automated download. Please verify local model file or configure environment variables."
    )


def load_prediction_model(model_path: Path | str | None = None) -> Any:
    """Load the Keras classification model for inference.

    Resolves the model path using resolve_model_path(), ensuring local
    availability or automated download as configured. Loads using
    compile=False to preserve model weights without optimizer overhead.

    Args:
        model_path: explicit path to the model file. If None, resolves
            using resolve_model_path().

    Returns:
        The loaded tf.keras.Model ready for inference.

    Raises:
        FileNotFoundError: if no model is available locally or via URL.
        ValueError: if SHA-256 verification fails.
        RuntimeError: if model download or deserialization fails.
    """
    import tensorflow as tf

    resolved_path = resolve_model_path(model_path=model_path)

    try:
        model = tf.keras.models.load_model(str(resolved_path), compile=False)
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {resolved_path}: {exc}"
        ) from exc
