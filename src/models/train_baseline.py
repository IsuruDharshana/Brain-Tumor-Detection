"""
train_baseline.py
-----------------
Phase 4 training script for the Brain Tumor MRI Classification project.

Trains the custom baseline CNN defined in ``src.models.baseline_cnn`` on the
Phase 3 tf.data pipeline and saves all artefacts required for the Phase 6
comparison against EfficientNetB0.

**Important**: the official Testing split is NOT loaded or evaluated here.
Test-set evaluation is intentionally reserved for Phase 6.

Usage (from the repository root, inside the virtual environment)::

    python -m src.models.train_baseline

Outputs
-------
    models/baseline_cnn.keras         — best checkpoint (by val_loss)
    results/baseline_cnn/
        model_summary.txt             — layer names, shapes, parameter counts
        training_history.csv          — per-epoch metrics
        training_curves.png           — accuracy and loss panels
        training_summary.json         — resulting metrics
        run_config.json               — experiment configuration (for reproducibility)
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Silence TensorFlow C++ log noise before importing TF.
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless WSL
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports — must be run from the repository root so the package is
# on sys.path (i.e.  python -m src.models.train_baseline).
# ---------------------------------------------------------------------------
from src.config import (
    DEFAULT_BATCH_SIZE,
    RANDOM_SEED,
    REPO_ROOT,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)
from src.data.data_pipeline import build_dataset, load_manifest
from src.models.baseline_cnn import build_baseline_cnn

# ---------------------------------------------------------------------------
# Training hyper-parameters (all constants, no argparse tuning in Phase 4)
# ---------------------------------------------------------------------------
LEARNING_RATE: float = 1e-3
MAX_EPOCHS: int = 30
BATCH_SIZE: int = DEFAULT_BATCH_SIZE  # 32

INPUT_SHAPE: tuple[int, int, int] = (224, 224, 3)
NUM_CLASSES: int = 4

# Callback settings
CHECKPOINT_MONITOR: str = "val_loss"
ES_PATIENCE: int = 5
LR_FACTOR: float = 0.5
LR_PATIENCE: int = 2
LR_MIN: float = 1e-6

# Output paths
MODELS_DIR: Path = REPO_ROOT / "models"
RESULTS_DIR: Path = REPO_ROOT / "results" / "baseline_cnn"
CHECKPOINT_PATH: Path = MODELS_DIR / "baseline_cnn.keras"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set Python, NumPy and TensorFlow random seeds for reproducibility.

    Note: full bit-for-bit determinism cannot be guaranteed across GPU
    hardware with TensorFlow's parallelised CUDA kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_gpu() -> tuple[bool, str]:
    """Return (gpu_available, gpu_name_or_message)."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return False, "No GPU detected"
    # Use the name of the first GPU.
    name = gpus[0].name  # e.g. /physical_device:GPU:0
    return True, name


def count_manifest_rows(manifest_path: Path) -> int:
    """Return the number of data rows (excluding the header) in a manifest."""
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return sum(1 for _ in reader)


def save_model_summary(model: tf.keras.Model, output_path: Path) -> None:
    """Write model.summary() to a text file."""
    lines: list[str] = []
    model.summary(print_fn=lambda line: lines.append(line))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_training_history(history: tf.keras.callbacks.History,
                          output_path: Path) -> None:
    """Write per-epoch metrics to a CSV file."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    # Collect whatever metric keys exist in the history dict.
    metric_keys = list(hist.keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch"] + metric_keys)
        for epoch in epochs:
            row = [epoch] + [hist[k][epoch - 1] for k in metric_keys]
            writer.writerow(row)


def save_training_curves(history: tf.keras.callbacks.History,
                         output_path: Path) -> None:
    """Plot training and validation accuracy/loss and save to PNG."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    # Resolve metric key names (Keras may use "accuracy" or
    # "sparse_categorical_accuracy" depending on the compile call).
    acc_key = "accuracy" if "accuracy" in hist else next(
        (k for k in hist if "accuracy" in k and not k.startswith("val_")), None
    )
    val_acc_key = f"val_{acc_key}" if acc_key else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Accuracy panel ---
    ax = axes[0]
    if acc_key and val_acc_key:
        ax.plot(epochs, hist[acc_key], label="Training accuracy", linewidth=1.5)
        ax.plot(epochs, hist[val_acc_key], label="Validation accuracy",
                linewidth=1.5, linestyle="--")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_xlim(1, max(epochs))

    # --- Loss panel ---
    ax = axes[1]
    ax.plot(epochs, hist["loss"], label="Training loss", linewidth=1.5)
    ax.plot(epochs, hist["val_loss"], label="Validation loss",
            linewidth=1.5, linestyle="--")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sparse Categorical Cross-entropy")
    ax.legend()
    ax.set_xlim(1, max(epochs))

    fig.suptitle("Baseline CNN — Training Curves", fontsize=13, y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_best_epoch(history: tf.keras.callbacks.History) -> int:
    """Return the 1-based epoch index with the lowest validation loss."""
    return int(np.argmin(history.history["val_loss"])) + 1


def get_param_counts(model: tf.keras.Model) -> dict[str, int]:
    """Return total, trainable and non-trainable parameter counts."""
    total = int(model.count_params())
    trainable = int(sum(
        np.prod(var.shape) for var in model.trainable_variables
    ))
    non_trainable = total - trainable
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
    }


def get_gpu_device_name() -> str | None:
    """Return a human-readable GPU name string, or None if unavailable."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None
    # tf.test.gpu_device_name() returns something like '/device:GPU:0'
    return tf.test.gpu_device_name() or gpus[0].name


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end Phase 4 baseline CNN training pipeline."""

    # -----------------------------------------------------------------------
    # Step 1 — Reproducibility seeds
    # -----------------------------------------------------------------------
    set_seeds(RANDOM_SEED)

    # -----------------------------------------------------------------------
    # Step 2 — GPU verification (hard fail if no GPU found)
    # -----------------------------------------------------------------------
    gpu_available, gpu_info = verify_gpu()
    if not gpu_available:
        print("=" * 60)
        print("ERROR: TensorFlow cannot see any GPU device.")
        print("Phase 4 training requires the RTX 3060 GPU.")
        print("Do not fall back to CPU. Check your CUDA/driver setup.")
        print("=" * 60)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3 — Validate manifest files exist
    # -----------------------------------------------------------------------
    for label, path in [("train", TRAIN_MANIFEST), ("validation", VAL_MANIFEST)]:
        if not path.is_file():
            print(f"ERROR: {label} manifest not found: {path}")
            print("Run 'python -m src.data.create_splits' first.")
            sys.exit(1)

    train_count = count_manifest_rows(TRAIN_MANIFEST)
    val_count = count_manifest_rows(VAL_MANIFEST)

    # -----------------------------------------------------------------------
    # Step 4 — Startup banner
    # -----------------------------------------------------------------------
    print()
    print("Baseline CNN Training")
    print("=====================")
    print(f"TensorFlow:        {tf.__version__}")
    print(f"Python:            {platform.python_version()}")
    print(f"GPU:               {gpu_info}")
    print(f"Training images:   {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Maximum epochs:    {MAX_EPOCHS}")
    print(f"Random seed:       {RANDOM_SEED}")
    print(f"Learning rate:     {LEARNING_RATE}")
    print()

    # -----------------------------------------------------------------------
    # Step 5 — Build tf.data datasets (NO test dataset)
    # -----------------------------------------------------------------------
    train_ds = build_dataset(TRAIN_MANIFEST, batch_size=BATCH_SIZE, training=True)
    val_ds = build_dataset(VAL_MANIFEST, batch_size=BATCH_SIZE, training=False)

    # -----------------------------------------------------------------------
    # Step 6 — Build and compile the model
    # -----------------------------------------------------------------------
    model = build_baseline_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # -----------------------------------------------------------------------
    # Step 7 — Print and save model summary
    # -----------------------------------------------------------------------
    model.summary()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model_summary(model, RESULTS_DIR / "model_summary.txt")
    print(f"\nModel summary saved → {RESULTS_DIR / 'model_summary.txt'}")

    # -----------------------------------------------------------------------
    # Step 8 — Callbacks
    # -----------------------------------------------------------------------
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_PATH),
            monitor=CHECKPOINT_MONITOR,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=CHECKPOINT_MONITOR,
            patience=ES_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=CHECKPOINT_MONITOR,
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=LR_MIN,
            verbose=1,
        ),
    ]

    # -----------------------------------------------------------------------
    # Step 9 — Training
    # -----------------------------------------------------------------------
    start_ts = datetime.now(tz=timezone.utc)
    start_time = time.monotonic()
    print(f"\nTraining started: {start_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    history = model.fit(
        train_ds,
        epochs=MAX_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    end_time = time.monotonic()
    end_ts = datetime.now(tz=timezone.utc)
    duration_seconds = end_time - start_time

    # -----------------------------------------------------------------------
    # Step 10 — Post-training analysis
    # -----------------------------------------------------------------------
    hist = history.history
    epochs_completed = len(hist["loss"])
    best_epoch = find_best_epoch(history)
    early_stopping_triggered = epochs_completed < MAX_EPOCHS

    # The best checkpoint was saved by ModelCheckpoint; EarlyStopping with
    # restore_best_weights=True also restores the best weights in memory.
    best_val_loss = float(hist["val_loss"][best_epoch - 1])

    # Resolve accuracy key name safely.
    acc_key = "accuracy" if "accuracy" in hist else next(
        (k for k in hist if "accuracy" in k and not k.startswith("val_")), None
    )
    val_acc_key = f"val_{acc_key}" if acc_key else None

    best_val_acc = float(hist[val_acc_key][best_epoch - 1]) if val_acc_key else None
    final_train_acc = float(hist[acc_key][-1]) if acc_key else None
    final_val_acc = float(hist[val_acc_key][-1]) if val_acc_key else None

    # Final learning rate: ReduceLROnPlateau stores it in history as 'lr'
    lr_key = next((k for k in hist if k.lower() == "lr"), None)
    final_lr = float(hist[lr_key][-1]) if lr_key else LEARNING_RATE

    # -----------------------------------------------------------------------
    # Step 11 — Save training history CSV
    # -----------------------------------------------------------------------
    history_path = RESULTS_DIR / "training_history.csv"
    save_training_history(history, history_path)
    print(f"\nTraining history saved → {history_path}")

    # -----------------------------------------------------------------------
    # Step 12 — Save training curves PNG
    # -----------------------------------------------------------------------
    curves_path = RESULTS_DIR / "training_curves.png"
    save_training_curves(history, curves_path)
    print(f"Training curves saved → {curves_path}")

    # -----------------------------------------------------------------------
    # Step 13 — Save training_summary.json
    # -----------------------------------------------------------------------
    param_counts = get_param_counts(model)
    gpu_device_name = get_gpu_device_name()

    training_summary = {
        "model_name": "baseline_cnn",
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
        "train_samples": train_count,
        "validation_samples": val_count,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_validation_loss": round(best_val_loss, 6),
        "best_validation_accuracy": round(best_val_acc, 6) if best_val_acc is not None else None,
        "final_training_accuracy": round(final_train_acc, 6) if final_train_acc is not None else None,
        "final_validation_accuracy": round(final_val_acc, 6) if final_val_acc is not None else None,
        "initial_learning_rate": LEARNING_RATE,
        "final_learning_rate": final_lr,
        "training_duration_seconds": round(duration_seconds, 1),
        "training_start_utc": start_ts.isoformat(),
        "training_end_utc": end_ts.isoformat(),
        **param_counts,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "gpu_detected": gpu_available,
        "gpu_device_name": gpu_device_name,
        "early_stopping_triggered": early_stopping_triggered,
        "best_model_path": str(CHECKPOINT_PATH),
    }

    summary_path = RESULTS_DIR / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(training_summary, fh, indent=2)
    print(f"Training summary saved → {summary_path}")

    # -----------------------------------------------------------------------
    # Step 14 — Save run_config.json
    # -----------------------------------------------------------------------
    run_config = {
        "seed": RANDOM_SEED,
        "image_size": [224, 224],
        "channels": 3,
        "normalization": "divide_by_255_float32_0_to_1",
        "augmentation": {
            "applied_to": "training split only",
            "rotation_degrees": 8,
            "zoom_factor": 0.08,
            "translation_factor": 0.05,
            "contrast_factor": 0.10,
            "flips": False,
        },
        "optimizer": "Adam",
        "loss": "SparseCategoricalCrossentropy",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "initial_learning_rate": LEARNING_RATE,
        "callbacks": {
            "ModelCheckpoint": {
                "monitor": CHECKPOINT_MONITOR,
                "save_best_only": True,
                "filepath": str(CHECKPOINT_PATH),
            },
            "EarlyStopping": {
                "monitor": CHECKPOINT_MONITOR,
                "patience": ES_PATIENCE,
                "restore_best_weights": True,
            },
            "ReduceLROnPlateau": {
                "monitor": CHECKPOINT_MONITOR,
                "factor": LR_FACTOR,
                "patience": LR_PATIENCE,
                "min_lr": LR_MIN,
            },
        },
        "architecture": {
            "name": "baseline_cnn",
            "blocks": [
                {"conv_filters": 32, "bn": True, "pool": "MaxPool2D"},
                {"conv_filters": 64, "bn": True, "pool": "MaxPool2D"},
                {"conv_filters": 128, "bn": True, "pool": "MaxPool2D"},
                {"conv_filters": 256, "bn": True, "pool": None},
            ],
            "head": "GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.4) → Dense(4, softmax)",
            "pretrained_weights": None,
        },
        "train_manifest": str(TRAIN_MANIFEST),
        "val_manifest": str(VAL_MANIFEST),
    }

    config_path = RESULTS_DIR / "run_config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)
    print(f"Run config saved      → {config_path}")

    # -----------------------------------------------------------------------
    # Step 15 — Sanity check: reload best model, one validation batch
    # -----------------------------------------------------------------------
    print("\n--- Sanity check: reloading best model ---")
    best_model = tf.keras.models.load_model(str(CHECKPOINT_PATH))
    val_batch_ds = build_dataset(VAL_MANIFEST, batch_size=BATCH_SIZE, training=False)
    sample_images, sample_labels = next(iter(val_batch_ds))
    predictions = best_model.predict(sample_images, verbose=0)

    pred_shape = tuple(predictions.shape)
    probs_finite = bool(np.all(np.isfinite(predictions)))
    prob_sums = predictions.sum(axis=1)
    probs_sum_to_one = bool(np.allclose(prob_sums, 1.0, atol=1e-5))
    pred_classes = predictions.argmax(axis=1)
    classes_valid = bool(np.all((pred_classes >= 0) & (pred_classes < NUM_CLASSES)))

    print(f"  Prediction shape:       {pred_shape}")
    print(f"  Probabilities finite:   {probs_finite}")
    print(f"  Probabilities sum ≈ 1:  {probs_sum_to_one}  "
          f"(min={prob_sums.min():.6f}, max={prob_sums.max():.6f})")
    print(f"  Class IDs in [0, {NUM_CLASSES-1}]:   {classes_valid}")

    if not (probs_finite and probs_sum_to_one and classes_valid):
        print("WARNING: sanity check failed — inspect the model outputs manually.")

    # -----------------------------------------------------------------------
    # Step 16 — Final training summary printout
    # -----------------------------------------------------------------------
    minutes, seconds = divmod(int(duration_seconds), 60)
    print()
    print("=" * 60)
    print("Training complete.")
    print("=" * 60)
    print(f"  Epochs completed:          {epochs_completed} / {MAX_EPOCHS}")
    print(f"  Early stopping triggered:  {early_stopping_triggered}")
    print(f"  Best epoch:                {best_epoch}")
    if best_val_acc is not None:
        print(f"  Best validation accuracy:  {best_val_acc:.4f}")
    print(f"  Best validation loss:      {best_val_loss:.4f}")
    if final_train_acc is not None:
        print(f"  Final training accuracy:   {final_train_acc:.4f}")
    if final_val_acc is not None:
        print(f"  Final validation accuracy: {final_val_acc:.4f}")
    print(f"  Final learning rate:       {final_lr:.2e}")
    print(f"  Training duration:         {minutes}m {seconds}s")
    print()
    print(f"  Model:   {CHECKPOINT_PATH}")
    print(f"  Results: {RESULTS_DIR}/")
    print()
    print("Test-set evaluation is reserved for Phase 6.")
    print("=" * 60)


if __name__ == "__main__":
    main()
