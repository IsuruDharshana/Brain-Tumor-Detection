"""
generate_gradcam_examples.py
----------------------------
Phase 6 reproducible Grad-CAM visualization generator for the selected
Baseline CNN model.

Deterministic selection rule:
  1. For each class (glioma, meningioma, notumor, pituitary):
     - Filter predictions where true_label == predicted_label for that class.
     - Sort deterministically by filepath (lexicographically).
     - Select the first filepath -> exactly one correct example per class.
  2. For misclassified examples:
     - Filter predictions where true_label_id != predicted_label_id.
     - Sort deterministically by filepath (lexicographically).
     - Select the first two filepaths -> up to two misclassified examples.

No training, no model modification, no full-test inference, no metric recalculation.
Computes Grad-CAM only on the selected images using the identified target layer
("block4_conv").

EDUCATIONAL / RESEARCH DISCLAIMER:
Grad-CAM highlights regions influencing the model's prediction. It is
not tumour segmentation and does not prove lesion location.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import CLASS_NAMES, REPO_ROOT
from src.data.preprocessing import load_and_preprocess_image
from src.explainability.gradcam import compute_gradcam, find_conv_layer, overlay_heatmap

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "baseline_cnn.keras"
DEFAULT_PREDICTIONS_CSV = REPO_ROOT / "results" / "final_evaluation" / "baseline_predictions.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "final_evaluation" / "gradcam"
TARGET_LAYER_NAME = "block4_conv"

DISCLAIMER_TEXT = (
    "Grad-CAM highlights regions influencing the model's prediction. "
    "It is not tumour segmentation and does not prove lesion location."
)


def resolve_image_path(path_str: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve an image path that may originate from Linux or Windows environments.

    If the path does not exist directly, attempts relative resolution
    starting from the 'data' directory against repo_root.
    """
    p = Path(path_str)
    if p.is_file():
        return p
    parts = p.parts
    if "data" in parts:
        idx = parts.index("data")
        candidate = repo_root.joinpath(*parts[idx:])
        if candidate.is_file():
            return candidate
    return p


def select_gradcam_examples(
    predictions_df: pd.DataFrame,
    class_names: tuple[str, ...] = CLASS_NAMES,
    max_misclassified: int = 2,
) -> list[dict[str, Any]]:
    """Select a deterministic set of correct and misclassified examples.

    Selection rules:
      - 1 correctly classified example per available class, sorted lexicographically by filepath.
      - Up to max_misclassified misclassified examples, sorted lexicographically by filepath.

    Args:
        predictions_df: DataFrame containing at least:
            filepath, true_label_id, true_label_name,
            predicted_label_id, predicted_label_name, confidence
        class_names: tuple of class names in fixed order.
        max_misclassified: maximum number of misclassified examples to select (default 2).

    Returns:
        List of dicts containing selected example metadata and assigned output filenames.
    """
    selected: list[dict[str, Any]] = []

    # 1. Exactly one correct example per class (lexicographically first filepath)
    for c_name in class_names:
        correct_subset = predictions_df[
            (predictions_df["true_label_name"] == c_name)
            & (predictions_df["predicted_label_name"] == c_name)
        ]
        if not correct_subset.empty:
            sorted_subset = correct_subset.sort_values(by="filepath", ascending=True)
            chosen = sorted_subset.iloc[0]
            selected.append({
                "filepath": str(chosen["filepath"]),
                "example_type": "correct",
                "true_label_id": int(chosen["true_label_id"]),
                "true_label_name": str(chosen["true_label_name"]),
                "predicted_label_id": int(chosen["predicted_label_id"]),
                "predicted_label_name": str(chosen["predicted_label_name"]),
                "confidence": float(chosen["confidence"]),
                "target_layer": TARGET_LAYER_NAME,
                "output_image": f"correct_{c_name}.png",
            })

    # 2. Up to max_misclassified misclassified examples (lexicographically first filepaths)
    misclassified_subset = predictions_df[
        predictions_df["true_label_id"] != predictions_df["predicted_label_id"]
    ]
    if not misclassified_subset.empty:
        sorted_mis = misclassified_subset.sort_values(by="filepath", ascending=True)
        chosen_mis = sorted_mis.head(max_misclassified)
        for i, (_, row) in enumerate(chosen_mis.iterrows(), start=1):
            selected.append({
                "filepath": str(row["filepath"]),
                "example_type": "misclassified",
                "true_label_id": int(row["true_label_id"]),
                "true_label_name": str(row["true_label_name"]),
                "predicted_label_id": int(row["predicted_label_id"]),
                "predicted_label_name": str(row["predicted_label_name"]),
                "confidence": float(row["confidence"]),
                "target_layer": TARGET_LAYER_NAME,
                "output_image": f"misclassified_{i:02d}.png",
            })

    return selected


def render_and_save_gradcam_panel(
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    example_meta: dict[str, Any],
    output_path: Path,
) -> None:
    """Render a 3-panel visualization: Original MRI | Grad-CAM Heatmap | Overlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    # Panel 1: Original MRI
    axes[0].imshow(image)
    axes[0].set_title("Original MRI", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Heatmap
    im2 = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Grad-CAM Heatmap ({example_meta['target_layer']})", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Overall title
    true_cls = example_meta["true_label_name"]
    pred_cls = example_meta["predicted_label_name"]
    conf = example_meta["confidence"]
    ex_type = example_meta["example_type"].capitalize()

    fig.suptitle(
        f"[{ex_type}] True: {true_cls}  |  Predicted: {pred_cls} (Confidence: {conf:.2%})\n"
        f"Educational/Research Visualization (Not Diagnostic)",
        fontsize=12,
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_gradcam_examples(
    model_path: Path = DEFAULT_MODEL_PATH,
    predictions_csv: Path = DEFAULT_PREDICTIONS_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_layer_name: str = TARGET_LAYER_NAME,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Generate and save Grad-CAM visualizations and metadata."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not predictions_csv.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions: {predictions_csv}")
    pred_df = pd.read_csv(predictions_csv)

    print("Selecting deterministic examples...")
    selected_examples = select_gradcam_examples(pred_df)
    print(f"Selected {len(selected_examples)} examples:")
    for ex in selected_examples:
        print(f"  - [{ex['example_type']:13s}] {ex['output_image']}: "
              f"True={ex['true_label_name']:10s} Pred={ex['predicted_label_name']:10s} "
              f"Conf={ex['confidence']:.4f}  Path={ex['filepath']}")

    print(f"\nLoading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    target_layer = find_conv_layer(model, preferred_name=target_layer_name)
    print(f"Target convolutional layer: {target_layer.name}")

    # Generate visualisations
    print("\nGenerating Grad-CAM overlays...")
    for ex in selected_examples:
        resolved = resolve_image_path(ex["filepath"], repo_root=repo_root)
        if not resolved.is_file():
            raise FileNotFoundError(f"Selected image file not found on disk: {resolved} (from {ex['filepath']})")

        img_tensor = load_and_preprocess_image(str(resolved))
        img_np = img_tensor.numpy()

        heatmap = compute_gradcam(
            model=model,
            image=img_np,
            class_idx=ex["predicted_label_id"],
            layer=target_layer,
        )
        overlay = overlay_heatmap(img_np, heatmap, alpha=0.4, colormap="jet")

        out_img_path = output_dir / ex["output_image"]
        render_and_save_gradcam_panel(
            image=img_np,
            heatmap=heatmap,
            overlay=overlay,
            example_meta=ex,
            output_path=out_img_path,
        )
        print(f"  Saved -> {out_img_path.name}")

    # Save metadata CSV
    meta_df = pd.DataFrame(selected_examples)
    csv_out_path = output_dir / "gradcam_examples.csv"
    meta_df.to_csv(csv_out_path, index=False)
    print(f"\nSaved metadata CSV -> {csv_out_path}")

    # Save summary JSON
    num_correct = sum(1 for ex in selected_examples if ex["example_type"] == "correct")
    num_misclassified = sum(1 for ex in selected_examples if ex["example_type"] == "misclassified")

    summary_data = {
        "selected_model": "Baseline CNN",
        "model_file": str(model_path.name),
        "target_layer": target_layer.name,
        "selection_rule": (
            "Deterministic lexicographic sort by filepath; exactly one correct example "
            "per class and up to two misclassified examples."
        ),
        "number_correct_examples": num_correct,
        "number_misclassified_examples": num_misclassified,
        "examples": selected_examples,
        "disclaimer": DISCLAIMER_TEXT,
    }

    json_out_path = output_dir / "gradcam_summary.json"
    with json_out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_data, fh, indent=2)
    print(f"Saved summary JSON -> {json_out_path}")

    return summary_data


def main() -> None:
    """Entry point for generating Grad-CAM examples."""
    print("=" * 70)
    print("PHASE 6 — REPRODUCIBLE GRAD-CAM EXAMPLES GENERATION")
    print("Selected Model: Baseline CNN (block4_conv)")
    print("=" * 70)

    summary = generate_gradcam_examples()

    print("\n" + "=" * 70)
    print("Grad-CAM generation complete.")
    print(f"Total examples: {len(summary['examples'])} "
          f"({summary['number_correct_examples']} correct, "
          f"{summary['number_misclassified_examples']} misclassified)")
    print(f"Disclaimer: {summary['disclaimer']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
