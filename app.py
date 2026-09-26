"""Streamlit interface for the Brain Tumor MRI Classification project.

The application performs single-image inference and Grad-CAM generation
entirely in memory. It is intended only as an educational/research demo.
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image
import streamlit as st

from src.app.gradcam_service import (
    DEFAULT_TARGET_LAYER_NAME,
    generate_gradcam_for_image,
)
from src.app.inference import (
    predict_image,
    preprocess_image,
    validate_mri_like_image,
)
from src.app.model_loader import load_prediction_model


st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def get_cached_model() -> Any:
    """Load and cache the trained Keras model for the app process."""
    return load_prediction_model()


CLINICAL_DISCLAIMER = (
    "This application is for educational and research purposes only. It is "
    "not a clinical diagnostic tool and must not be used for medical "
    "decision-making."
)

CONFIDENCE_DISCLAIMER = (
    "Prediction confidence is the model's softmax output and should not be "
    "interpreted as clinical certainty."
)

GRADCAM_DESCRIPTION = (
    "Grad-CAM highlights image regions that influenced the model's prediction."
)

GRADCAM_DISCLAIMER = (
    "It is not tumour segmentation and does not prove lesion location."
)

PRIVACY_NOTE = (
    "Uploaded images are processed for the current app session and are not "
    "intentionally stored by this application."
)

INPUT_VALIDATION_LIMITATION = (
    "The application performs only a basic input plausibility check and cannot "
    "verify medical image authenticity."
)

# Preserve the model's canonical class order in every probability display.
CLASS_DISPLAY_ORDER = ("Glioma", "Meningioma", "No Tumor", "Pituitary")


def render_sidebar() -> None:
    """Render compact supporting information without dominating the workflow."""
    with st.sidebar:
        st.header("Project Information")
        with st.expander("About the Model"):
            st.markdown(
                """
                **Selected Architecture**  
                Baseline Custom CNN

                **Input**  
                224 × 224 RGB

                **Classes**  
                Glioma  
                Meningioma  
                No Tumor  
                Pituitary

                **Official Test Accuracy**  
                85.13%

                **Macro F1**  
                84.95%

                **Grad-CAM Target Layer**  
                `block4_conv`
                """
            )

        st.caption(f"Privacy: {PRIVACY_NOTE}")


def render_footer() -> None:
    """Render the closing educational and privacy notices."""
    st.markdown("---")
    st.caption(CLINICAL_DISCLAIMER)
    st.caption(f"Privacy: {PRIVACY_NOTE}")


def render_probabilities(probabilities: dict[str, float]) -> None:
    """Render all four class probabilities in their fixed canonical order."""
    st.subheader("Class Probabilities")
    for class_name in CLASS_DISPLAY_ORDER:
        probability = float(probabilities[class_name])
        st.progress(
            probability,
            text=f"{class_name}: {probability * 100.0:.2f}%",
        )


def main() -> None:
    render_sidebar()

    st.title("Brain Tumor MRI Classification")
    st.subheader("Educational / Research Demonstration")
    st.warning(CLINICAL_DISCLAIMER)

    st.subheader("1. Upload MRI")
    st.write("Upload a brain MRI image in JPG, JPEG, or PNG format.")
    uploaded_file = st.file_uploader(
        "Upload a Brain MRI Image",
        type=["jpg", "jpeg", "png"],
        help="Supported file types: JPG, JPEG, and PNG.",
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to preview it. Analysis starts only after you select "
            "Analyze Image."
        )
        render_footer()
        return

    try:
        uploaded_bytes = uploaded_file.getvalue()
        pil_image = Image.open(io.BytesIO(uploaded_bytes))
        pil_image.load()
    except Exception:
        st.error(
            "Unable to read this image. Please upload a valid JPG, JPEG, or PNG file."
        )
        render_footer()
        return

    preview_column, result_column = st.columns(2, gap="large")

    with preview_column:
        st.subheader("2. MRI Image Preview")
        st.image(
            pil_image,
            caption=uploaded_file.name,
            use_column_width=True,
        )

    with result_column:
        st.subheader("3. Analyze Image")
        st.write("Run the trained model and generate an explanation for this image.")
        analyze_button = st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True,
        )

    if not analyze_button:
        render_footer()
        return

    try:
        validation_result = validate_mri_like_image(pil_image)
    except Exception:
        with result_column:
            st.error("The image could not be validated. Please try another image.")
        render_footer()
        return

    if not validation_result["is_plausible_mri"]:
        with result_column:
            st.warning(
                "This image does not appear to be a grayscale brain MRI scan. "
                "Please upload a valid brain MRI image."
            )
            st.caption(INPUT_VALIDATION_LIMITATION)
        render_footer()
        return

    try:
        model = get_cached_model()
    except Exception:
        with result_column:
            st.error(
                "Model is currently unavailable. Please try again later."
            )
        render_footer()
        return

    with st.spinner("Analyzing the image..."):
        try:
            preprocessed_tensor = preprocess_image(uploaded_bytes)
            prediction_result = predict_image(model, preprocessed_tensor)
        except Exception:
            with result_column:
                st.error(
                    "The image could not be analyzed. Please try again with a valid image."
                )
            render_footer()
            return

        try:
            gradcam_result = generate_gradcam_for_image(
                model=model,
                preprocessed_image=preprocessed_tensor,
                target_class_id=prediction_result["predicted_class_id"],
                target_layer_name=DEFAULT_TARGET_LAYER_NAME,
            )
        except Exception:
            gradcam_result = {"success": False}

    pred_display = prediction_result["predicted_display_name"]
    confidence_pct = prediction_result["confidence"] * 100.0

    with result_column:
        st.subheader("Model Prediction")
        prediction_metric, confidence_metric = st.columns(2)
        with prediction_metric:
            st.metric("Model Prediction", pred_display)
        with confidence_metric:
            st.metric("Prediction Confidence", f"{confidence_pct:.1f}%")
        st.caption(CONFIDENCE_DISCLAIMER)

    st.markdown("---")
    render_probabilities(prediction_result["display_probabilities"])

    st.markdown("---")
    st.header("Grad-CAM Explanation")
    st.write(GRADCAM_DESCRIPTION)

    if gradcam_result.get("success"):
        original_column, heatmap_column, overlay_column = st.columns(3)
        with original_column:
            st.image(
                gradcam_result["original_image"],
                caption="Original MRI",
                use_column_width=True,
                clamp=True,
            )
        with heatmap_column:
            st.image(
                gradcam_result["heatmap"],
                caption="Grad-CAM Heatmap",
                use_column_width=True,
                clamp=True,
            )
        with overlay_column:
            st.image(
                gradcam_result["overlay"],
                caption="Overlay",
                use_column_width=True,
                clamp=True,
            )
    else:
        st.warning(
            "Prediction completed, but the Grad-CAM explanation could not be generated."
        )

    st.info(GRADCAM_DISCLAIMER)
    render_footer()


if __name__ == "__main__":
    main()
