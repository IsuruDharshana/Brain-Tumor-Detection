"""
train_efficientnet.py
---------------------
Phase 5 training script — EfficientNetB0 transfer learning for the Brain
Tumor MRI Classification project.

Two-stage transfer learning::

    Stage 1 — Frozen backbone
        EfficientNetB0 backbone is fully frozen.
        Only the custom classifier head is trained.
        Optimizer: Adam @ 1e-3.
        Max epochs: 15 (EarlyStopping patience 4).

    Stage 2 — Controlled fine-tuning
        Top ~30 backbone layers are unfrozen (BatchNorm layers stay frozen).
        Very small learning rate: Adam @ 1e-5.
        Max epochs: 15 (EarlyStopping patience 4).

Model selection:
    The final model is selected by comparing the best validation loss of the
    two stages. The selected model is saved to models/efficientnet_b0.keras.

IMPORTANT — Test-set rule:
    The official Testing split (data/processed/test_manifest.csv) is NEVER
    loaded, evaluated or inspected in this script. Test evaluation is
    reserved for Phase 6.

Preprocessing:
    Phase 3 pipeline delivers float32 images in [0, 1].
    The model includes an internal Rescaling(255.0) adapter so the backbone
    receives the [0, 255] range it was pretrained on. The data pipeline is
    unchanged.

Usage (from the repository root, inside the virtual environment)::

    python -m src.models.train_efficientnet

Outputs:
    models/efficientnet_b0_frozen.keras     — best Stage 1 checkpoint
    models/efficientnet_b0_finetuned.keras  — best Stage 2 checkpoint
    models/efficientnet_b0.keras            — final selected model (copy)
    results/efficientnet_b0/
        model_summary.txt
        frozen_history.csv
        finetune_history.csv
        training_curves.png
        training_summary.json
        run_config.json
        stage_comparison.json
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Suppress TensorFlow C++ log noise before importing TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless WSL
import matplotlib.pyplot as plt

from src.config import (
    DEFAULT_BATCH_SIZE,
    LABEL_TO_ID,
    RANDOM_SEED,
    REPO_ROOT,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)
from src.data.data_pipeline import build_dataset
from src.models.efficientnet_b0 import (
    build_efficientnet_b0,
    count_trainable_layers,
    freeze_backbone,
    unfreeze_top_layers,
)

# ---------------------------------------------------------------------------
# Hyper-parameters (fixed; no tuning in Phase 5)
# ---------------------------------------------------------------------------
BATCH_SIZE: int = DEFAULT_BATCH_SIZE           # 32
INPUT_SHAPE: tuple[int, int, int] = (224, 224, 3)
NUM_CLASSES: int = 4
DROPOUT_RATE: float = 0.3

# Stage 1 — frozen backbone
S1_LR: float = 1e-3
S1_MAX_EPOCHS: int = 15
S1_ES_PATIENCE: int = 4
S1_LR_PATIENCE: int = 2
S1_LR_FACTOR: float = 0.5
S1_LR_MIN: float = 1e-6

# Stage 2 — fine-tuning
S2_LR: float = 1e-5
S2_MAX_EPOCHS: int = 15
S2_ES_PATIENCE: int = 4
S2_LR_PATIENCE: int = 2
S2_LR_FACTOR: float = 0.5
S2_LR_MIN: float = 1e-7
S2_UNFREEZE_LAYERS: int = 30   # top N backbone layers to unfreeze

CHECKPOINT_MONITOR: str = "val_loss"

# Output paths
MODELS_DIR: Path = REPO_ROOT / "models"
RESULTS_DIR: Path = REPO_ROOT / "results" / "efficientnet_b0"
S1_CHECKPOINT: Path = MODELS_DIR / "efficientnet_b0_frozen.keras"
S2_CHECKPOINT: Path = MODELS_DIR / "efficientnet_b0_finetuned.keras"
FINAL_MODEL: Path = MODELS_DIR / "efficientnet_b0.keras"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set Python, NumPy and TensorFlow random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_gpu() -> tuple[bool, str]:
    """Return (gpu_available, gpu_name_or_message)."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return False, "No GPU detected"
    return True, gpus[0].name


def count_manifest_rows(manifest_path: Path) -> int:
    """Count data rows (excluding header) in a manifest CSV."""
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def save_model_summary(model: tf.keras.Model, output_path: Path) -> None:
    """Write model.summary() to a text file."""
    lines: list[str] = []
    model.summary(print_fn=lambda line: lines.append(line))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_history_csv(
    history: tf.keras.callbacks.History,
    output_path: Path,
) -> None:
    """Write per-epoch metrics to a CSV file."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    metric_keys = list(hist.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch"] + metric_keys)
        for epoch in epochs:
            writer.writerow([epoch] + [hist[k][epoch - 1] for k in metric_keys])


def find_best_epoch(history: tf.keras.callbacks.History) -> int:
    """Return 1-based epoch with lowest validation loss."""
    return int(np.argmin(history.history["val_loss"])) + 1


def resolve_metric_keys(
    history: tf.keras.callbacks.History,
) -> tuple[str | None, str | None]:
    """Return (train_acc_key, val_acc_key) accounting for different Keras naming."""
    hist = history.history
    acc_key = "accuracy" if "accuracy" in hist else next(
        (k for k in hist if "accuracy" in k and not k.startswith("val_")), None
    )
    val_acc_key = f"val_{acc_key}" if acc_key else None
    return acc_key, val_acc_key


def get_param_counts(model: tf.keras.Model) -> dict[str, int]:
    """Return total, trainable and non-trainable parameter counts."""
    import numpy as np
    total = int(model.count_params())
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
    }


def get_final_lr(history: tf.keras.callbacks.History,
                  fallback: float) -> float:
    """Extract the final learning rate from history, or return fallback."""
    lr_key = next((k for k in history.history if k.lower() == "lr"), None)
    if lr_key:
        return float(history.history[lr_key][-1])
    return fallback


def make_callbacks(
    checkpoint_path: Path,
    es_patience: int,
    lr_patience: int,
    lr_factor: float,
    lr_min: float,
) -> list[tf.keras.callbacks.Callback]:
    """Build the standard callback list for a training stage."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=CHECKPOINT_MONITOR,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=CHECKPOINT_MONITOR,
            patience=es_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=CHECKPOINT_MONITOR,
            factor=lr_factor,
            patience=lr_patience,
            min_lr=lr_min,
            verbose=1,
        ),
    ]


def save_training_curves(
    s1_history: tf.keras.callbacks.History,
    s2_history: tf.keras.callbacks.History,
    output_path: Path,
) -> None:
    """Generate a 2×2 panel figure showing Stage 1 and Stage 2 curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "EfficientNetB0 — Training Curves (Phase 5)",
        fontsize=14,
        y=1.01,
    )

    for stage_idx, (hist_obj, stage_label) in enumerate(
        [(s1_history, "Stage 1 — Frozen backbone"),
         (s2_history, "Stage 2 — Fine-tuning")]
    ):
        hist = hist_obj.history
        n_epochs = len(hist["loss"])
        epochs = range(1, n_epochs + 1)
        acc_key, val_acc_key = resolve_metric_keys(hist_obj)

        # Accuracy
        ax_acc = axes[stage_idx][0]
        if acc_key and val_acc_key:
            ax_acc.plot(epochs, hist[acc_key],
                        label="Training", linewidth=1.5)
            ax_acc.plot(epochs, hist[val_acc_key],
                        label="Validation", linewidth=1.5, linestyle="--")
        ax_acc.set_title(f"{stage_label}\nAccuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        if n_epochs > 1:
            ax_acc.set_xlim(1, n_epochs)

        # Loss
        ax_loss = axes[stage_idx][1]
        ax_loss.plot(epochs, hist["loss"],
                     label="Training", linewidth=1.5)
        ax_loss.plot(epochs, hist["val_loss"],
                     label="Validation", linewidth=1.5, linestyle="--")
        ax_loss.set_title(f"{stage_label}\nLoss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Sparse Categorical Cross-entropy")
        ax_loss.legend()
        if n_epochs > 1:
            ax_loss.set_xlim(1, n_epochs)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end Phase 5 EfficientNetB0 training pipeline."""

    # -----------------------------------------------------------------------
    # Step 1 — Seeds
    # -----------------------------------------------------------------------
    set_seeds(RANDOM_SEED)

    # -----------------------------------------------------------------------
    # Step 2 — GPU verification (hard fail if no GPU)
    # -----------------------------------------------------------------------
    gpu_available, gpu_info = verify_gpu()
    if not gpu_available:
        print("=" * 60)
        print("ERROR: TensorFlow cannot see any GPU device.")
        print("Phase 5 training requires the RTX 3060 GPU.")
        print("Do not fall back to CPU. Check your CUDA/driver setup.")
        print("=" * 60)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3 — Validate manifests (TRAIN + VAL only; test manifest NOT loaded)
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
    print("EfficientNetB0 Transfer Learning — Phase 5")
    print("=" * 50)
    print(f"TensorFlow:        {tf.__version__}")
    print(f"Python:            {platform.python_version()}")
    print(f"GPU:               {gpu_info}")
    print(f"Training images:   {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Random seed:       {RANDOM_SEED}")
    print()
    print("Preprocessing: Phase 3 pipeline delivers [0,1] float32.")
    print("  Model adapter: Rescaling(255.0) inside model head.")
    print("  EfficientNetB0 backbone receives [0,255] as expected.")
    print("  Data pipeline unchanged. No double normalisation.")
    print()

    # -----------------------------------------------------------------------
    # Step 5 — Build tf.data datasets (NO test dataset)
    # -----------------------------------------------------------------------
    train_ds = build_dataset(TRAIN_MANIFEST, batch_size=BATCH_SIZE, training=True)
    val_ds = build_dataset(VAL_MANIFEST, batch_size=BATCH_SIZE, training=False)

    # -----------------------------------------------------------------------
    # Step 6 — Create output directories
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 7 — Build Stage 1 model (frozen backbone)
    # -----------------------------------------------------------------------
    print("Building EfficientNetB0 model (weights='imagenet') …")
    model = build_efficientnet_b0(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        weights="imagenet",
        dropout_rate=DROPOUT_RATE,
    )

    # Freeze backbone — only head trains in Stage 1.
    freeze_backbone(model)

    s1_layer_counts = count_trainable_layers(model)
    s1_param_counts = get_param_counts(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=S1_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print()
    print("--- Stage 1: Frozen backbone ---")
    print(f"  Trainable params:  {s1_param_counts['trainable_parameters']:,}")
    print(f"  Frozen params:     {s1_param_counts['non_trainable_parameters']:,}")
    print(f"  Trainable layers:  {s1_layer_counts['trainable']}")
    print(f"  Frozen layers:     {s1_layer_counts['frozen']}")
    print(f"  Learning rate:     {S1_LR}")
    print(f"  Max epochs:        {S1_MAX_EPOCHS}")
    print()

    # Save model summary (Stage 1 config).
    save_model_summary(model, RESULTS_DIR / "model_summary.txt")

    # -----------------------------------------------------------------------
    # Step 8 — Stage 1 training
    # -----------------------------------------------------------------------
    s1_start = time.monotonic()
    s1_start_ts = datetime.now(tz=timezone.utc)

    s1_history = model.fit(
        train_ds,
        epochs=S1_MAX_EPOCHS,
        validation_data=val_ds,
        callbacks=make_callbacks(
            S1_CHECKPOINT, S1_ES_PATIENCE, S1_LR_PATIENCE, S1_LR_FACTOR, S1_LR_MIN
        ),
        verbose=1,
    )

    s1_duration = time.monotonic() - s1_start
    s1_epochs = len(s1_history.history["loss"])
    s1_best_epoch = find_best_epoch(s1_history)
    s1_best_val_loss = float(s1_history.history["val_loss"][s1_best_epoch - 1])
    s1_acc_key, s1_val_acc_key = resolve_metric_keys(s1_history)
    s1_best_val_acc = (
        float(s1_history.history[s1_val_acc_key][s1_best_epoch - 1])
        if s1_val_acc_key else None
    )
    # Read the actual final LR from the optimizer before the model is
    # reloaded.  This is the true post-ReduceLROnPlateau value and is more
    # reliable than parsing the history "lr" key (which some Keras versions
    # do not log by default).
    s1_final_lr = float(
        tf.keras.backend.get_value(model.optimizer.learning_rate)
    )
    s1_es_triggered = s1_epochs < S1_MAX_EPOCHS

    _s1_acc_text = f"{s1_best_val_acc:.4f}" if s1_best_val_acc is not None else "N/A"
    print()
    print(f"Stage 1 complete. Epochs: {s1_epochs}  "
          f"Best epoch: {s1_best_epoch}  "
          f"Best val_loss: {s1_best_val_loss:.4f}  "
          f"Best val_acc: {_s1_acc_text}")
    print(f"Stage 1 duration: {s1_duration:.0f}s")

    save_history_csv(s1_history, RESULTS_DIR / "frozen_history.csv")
    print(f"Stage 1 history saved → {RESULTS_DIR / 'frozen_history.csv'}")

    # -----------------------------------------------------------------------
    # Step 9 — Reload best Stage 1 checkpoint before fine-tuning
    # -----------------------------------------------------------------------
    print()
    print("Reloading best Stage 1 checkpoint for fine-tuning …")
    model = tf.keras.models.load_model(str(S1_CHECKPOINT))

    # -----------------------------------------------------------------------
    # Step 10 — Configure Stage 2 (fine-tuning)
    # -----------------------------------------------------------------------
    unfreeze_top_layers(
        model,
        num_layers=S2_UNFREEZE_LAYERS,
        keep_bn_frozen=True,
    )

    s2_layer_counts = count_trainable_layers(model)
    s2_param_counts = get_param_counts(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=S2_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print()
    print("--- Stage 2: Fine-tuning ---")
    print(f"  Unfrozen backbone layers: {S2_UNFREEZE_LAYERS}")
    print(f"  BatchNorm layers:         kept frozen")
    print(f"  Trainable params:  {s2_param_counts['trainable_parameters']:,}")
    print(f"  Frozen params:     {s2_param_counts['non_trainable_parameters']:,}")
    print(f"  Trainable layers:  {s2_layer_counts['trainable']}")
    print(f"  Frozen layers:     {s2_layer_counts['frozen']}")
    print(f"  Learning rate:     {S2_LR}")
    print(f"  Max epochs:        {S2_MAX_EPOCHS}")
    print()

    # -----------------------------------------------------------------------
    # Step 11 — Stage 2 training
    # -----------------------------------------------------------------------
    s2_start = time.monotonic()

    s2_history = model.fit(
        train_ds,
        epochs=S2_MAX_EPOCHS,
        validation_data=val_ds,
        callbacks=make_callbacks(
            S2_CHECKPOINT, S2_ES_PATIENCE, S2_LR_PATIENCE, S2_LR_FACTOR, S2_LR_MIN
        ),
        verbose=1,
    )

    s2_duration = time.monotonic() - s2_start
    s2_epochs = len(s2_history.history["loss"])
    s2_best_epoch = find_best_epoch(s2_history)
    s2_best_val_loss = float(s2_history.history["val_loss"][s2_best_epoch - 1])
    s2_acc_key, s2_val_acc_key = resolve_metric_keys(s2_history)
    s2_best_val_acc = (
        float(s2_history.history[s2_val_acc_key][s2_best_epoch - 1])
        if s2_val_acc_key else None
    )
    # Read the actual final LR from the Stage 2 optimizer immediately after
    # training ends, before any further model operations.
    s2_final_lr = float(
        tf.keras.backend.get_value(model.optimizer.learning_rate)
    )
    s2_es_triggered = s2_epochs < S2_MAX_EPOCHS

    total_duration = s2_start + s2_duration - (time.monotonic() - s1_duration - s1_start)
    total_duration = s1_duration + s2_duration

    _s2_acc_text = f"{s2_best_val_acc:.4f}" if s2_best_val_acc is not None else "N/A"
    print()
    print(f"Stage 2 complete. Epochs: {s2_epochs}  "
          f"Best epoch: {s2_best_epoch}  "
          f"Best val_loss: {s2_best_val_loss:.4f}  "
          f"Best val_acc: {_s2_acc_text}")
    print(f"Stage 2 duration: {s2_duration:.0f}s")

    save_history_csv(s2_history, RESULTS_DIR / "finetune_history.csv")
    print(f"Stage 2 history saved → {RESULTS_DIR / 'finetune_history.csv'}")

    # -----------------------------------------------------------------------
    # Step 12 — Model selection (validation loss only; NO test set)
    # -----------------------------------------------------------------------
    if s2_best_val_loss <= s1_best_val_loss:
        selected_stage = 2
        selected_checkpoint = S2_CHECKPOINT
        selected_val_loss = s2_best_val_loss
        selected_val_acc = s2_best_val_acc
    else:
        selected_stage = 1
        selected_checkpoint = S1_CHECKPOINT
        selected_val_loss = s1_best_val_loss
        selected_val_acc = s1_best_val_acc

    print()
    print(f"Model selection (by val_loss):")
    print(f"  Stage 1 best val_loss: {s1_best_val_loss:.4f}")
    print(f"  Stage 2 best val_loss: {s2_best_val_loss:.4f}")
    print(f"  → Selected: Stage {selected_stage}")

    # Copy selected checkpoint to final model path.
    shutil.copy2(str(selected_checkpoint), str(FINAL_MODEL))
    print(f"  Final model saved → {FINAL_MODEL}")

    # -----------------------------------------------------------------------
    # Step 13 — Training curves
    # -----------------------------------------------------------------------
    curves_path = RESULTS_DIR / "training_curves.png"
    save_training_curves(s1_history, s2_history, curves_path)
    print(f"Training curves saved → {curves_path}")

    # -----------------------------------------------------------------------
    # Step 14 — stage_comparison.json
    # -----------------------------------------------------------------------
    stage_comparison = {
        "stage_1_frozen": {
            "epochs_completed": s1_epochs,
            "best_epoch": s1_best_epoch,
            "best_validation_loss": round(s1_best_val_loss, 6),
            "best_validation_accuracy": round(s1_best_val_acc, 6) if s1_best_val_acc is not None else None,
            "final_learning_rate": s1_final_lr,
            "early_stopping_triggered": s1_es_triggered,
            "duration_seconds": round(s1_duration, 1),
        },
        "stage_2_finetuned": {
            "epochs_completed": s2_epochs,
            "best_epoch": s2_best_epoch,
            "best_validation_loss": round(s2_best_val_loss, 6),
            "best_validation_accuracy": round(s2_best_val_acc, 6) if s2_best_val_acc is not None else None,
            "final_learning_rate": s2_final_lr,
            "early_stopping_triggered": s2_es_triggered,
            "duration_seconds": round(s2_duration, 1),
        },
        "selected_stage": selected_stage,
        "selected_validation_loss": round(selected_val_loss, 6),
        "selected_validation_accuracy": round(selected_val_acc, 6) if selected_val_acc is not None else None,
        "selection_criterion": "lowest val_loss",
        "test_set_evaluated": False,
    }

    comparison_path = RESULTS_DIR / "stage_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as fh:
        json.dump(stage_comparison, fh, indent=2)
    print(f"Stage comparison saved → {comparison_path}")

    # -----------------------------------------------------------------------
    # Step 15 — training_summary.json
    # -----------------------------------------------------------------------
    gpu_device_name = tf.test.gpu_device_name() or gpu_info

    training_summary = {
        "model_name": "efficientnet_b0_transfer",
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "gpu_detected": gpu_available,
        "gpu_device_name": gpu_device_name,
        "train_samples": train_count,
        "validation_samples": val_count,
        "batch_size": BATCH_SIZE,
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
        "class_mapping": LABEL_TO_ID,
        "preprocessing_strategy": (
            "Phase 3 pipeline delivers float32 [0,1]. "
            "Model-side Rescaling(255.0) adapter converts to [0,255] "
            "before EfficientNetB0 backbone (which has its own internal "
            "Rescaling(1/255)). No double normalisation."
        ),
        "stage_1_frozen": {
            "optimizer": "Adam",
            "initial_learning_rate": S1_LR,
            "max_epochs": S1_MAX_EPOCHS,
            "epochs_completed": s1_epochs,
            "best_epoch": s1_best_epoch,
            "best_validation_loss": round(s1_best_val_loss, 6),
            "best_validation_accuracy": round(s1_best_val_acc, 6) if s1_best_val_acc is not None else None,
            "final_learning_rate": s1_final_lr,
            "early_stopping_triggered": s1_es_triggered,
            "trainable_parameters": s1_param_counts["trainable_parameters"],
            "frozen_parameters": s1_param_counts["non_trainable_parameters"],
            "trainable_layers": s1_layer_counts["trainable"],
            "frozen_layers": s1_layer_counts["frozen"],
            "duration_seconds": round(s1_duration, 1),
        },
        "stage_2_finetuned": {
            "optimizer": "Adam",
            "initial_learning_rate": S2_LR,
            "max_epochs": S2_MAX_EPOCHS,
            "epochs_completed": s2_epochs,
            "best_epoch": s2_best_epoch,
            "best_validation_loss": round(s2_best_val_loss, 6),
            "best_validation_accuracy": round(s2_best_val_acc, 6) if s2_best_val_acc is not None else None,
            "final_learning_rate": s2_final_lr,
            "early_stopping_triggered": s2_es_triggered,
            "unfrozen_backbone_layers": S2_UNFREEZE_LAYERS,
            "batchnorm_kept_frozen": True,
            "trainable_parameters": s2_param_counts["trainable_parameters"],
            "frozen_parameters": s2_param_counts["non_trainable_parameters"],
            "trainable_layers": s2_layer_counts["trainable"],
            "frozen_layers": s2_layer_counts["frozen"],
            "duration_seconds": round(s2_duration, 1),
        },
        "selected_stage": selected_stage,
        "selected_validation_loss": round(selected_val_loss, 6),
        "selected_validation_accuracy": round(selected_val_acc, 6) if selected_val_acc is not None else None,
        "total_training_duration_seconds": round(total_duration, 1),
        "best_model_path": str(FINAL_MODEL),
        "test_set_evaluated": False,
    }

    summary_path = RESULTS_DIR / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(training_summary, fh, indent=2)
    print(f"Training summary saved → {summary_path}")

    # -----------------------------------------------------------------------
    # Step 16 — run_config.json
    # -----------------------------------------------------------------------
    run_config = {
        "seed": RANDOM_SEED,
        "image_size": [224, 224],
        "channels": 3,
        "preprocessing": {
            "data_pipeline_range": "[0, 1] float32",
            "adapter": "Rescaling(255.0) inside model",
            "backbone_input_range": "[0, 255] float32",
            "backbone_internal_rescaling": "Rescaling(1/255) inside EfficientNetB0",
        },
        "augmentation": {
            "applied_to": "training split only",
            "rotation_degrees": 8,
            "zoom_factor": 0.08,
            "translation_factor": 0.05,
            "contrast_factor": 0.10,
            "flips": False,
        },
        "stage_1": {
            "description": "frozen backbone — head only",
            "optimizer": "Adam",
            "learning_rate": S1_LR,
            "max_epochs": S1_MAX_EPOCHS,
            "es_patience": S1_ES_PATIENCE,
            "lr_patience": S1_LR_PATIENCE,
            "lr_factor": S1_LR_FACTOR,
            "lr_min": S1_LR_MIN,
            "checkpoint": str(S1_CHECKPOINT),
        },
        "stage_2": {
            "description": "controlled fine-tuning",
            "unfrozen_backbone_layers": S2_UNFREEZE_LAYERS,
            "bn_frozen": True,
            "optimizer": "Adam",
            "learning_rate": S2_LR,
            "max_epochs": S2_MAX_EPOCHS,
            "es_patience": S2_ES_PATIENCE,
            "lr_patience": S2_LR_PATIENCE,
            "lr_factor": S2_LR_FACTOR,
            "lr_min": S2_LR_MIN,
            "checkpoint": str(S2_CHECKPOINT),
        },
        "batch_size": BATCH_SIZE,
        "loss": "SparseCategoricalCrossentropy",
        "num_classes": NUM_CLASSES,
        "final_model": str(FINAL_MODEL),
    }

    config_path = RESULTS_DIR / "run_config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)
    print(f"Run config saved → {config_path}")

    # -----------------------------------------------------------------------
    # Step 17 — Sanity check on ONE validation batch (no test set)
    # -----------------------------------------------------------------------
    print()
    print("--- Sanity check: reloading final model ---")
    final_model = tf.keras.models.load_model(str(FINAL_MODEL))
    val_batch_ds = build_dataset(VAL_MANIFEST, batch_size=BATCH_SIZE, training=False)
    sample_images, sample_labels = next(iter(val_batch_ds))
    predictions = final_model.predict(sample_images, verbose=0)

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
    print(f"  Class IDs in [0, {NUM_CLASSES - 1}]:   {classes_valid}")

    if not (probs_finite and probs_sum_to_one and classes_valid):
        print("WARNING: sanity check failed — inspect model outputs manually.")

    # -----------------------------------------------------------------------
    # Step 18 — Final summary printout
    # -----------------------------------------------------------------------
    total_min, total_sec = divmod(int(total_duration), 60)
    _s1_acc_summary = f"{s1_best_val_acc:.4f}" if s1_best_val_acc is not None else "N/A"
    _s2_acc_summary = f"{s2_best_val_acc:.4f}" if s2_best_val_acc is not None else "N/A"
    _sel_acc_summary = f"{selected_val_acc:.4f}" if selected_val_acc is not None else "N/A"
    print()
    print("=" * 60)
    print("Phase 5 training complete.")
    print("=" * 60)
    print(f"  Stage 1 epochs:            {s1_epochs} / {S1_MAX_EPOCHS}")
    print(f"  Stage 1 best val_acc:      {_s1_acc_summary}")
    print(f"  Stage 1 best val_loss:     {s1_best_val_loss:.4f}")
    print(f"  Stage 2 epochs:            {s2_epochs} / {S2_MAX_EPOCHS}")
    print(f"  Stage 2 best val_acc:      {_s2_acc_summary}")
    print(f"  Stage 2 best val_loss:     {s2_best_val_loss:.4f}")
    print(f"  Selected stage:            Stage {selected_stage}")
    print(f"  Selected val_acc:          {_sel_acc_summary}")
    print(f"  Selected val_loss:         {selected_val_loss:.4f}")
    print(f"  Total runtime:             {total_min}m {total_sec}s")
    print()
    print(f"  Final model:  {FINAL_MODEL}")
    print(f"  Results:      {RESULTS_DIR}/")
    print()
    print("  Test-set result: NOT EVALUATED — reserved for Phase 6")
    print("=" * 60)


if __name__ == "__main__":
    main()
