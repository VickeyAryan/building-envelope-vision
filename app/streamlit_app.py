"""Streamlit demo: upload a building facade photo, get a segmentation overlay
plus computed WWR / wall-tone / shading-coverage metrics.

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as TF

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, PALETTE  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402
from model import load_checkpoint  # noqa: E402

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pt"
IMAGE_SIZE = 384


@st.cache_resource
def get_model():
    if not CHECKPOINT_PATH.exists():
        return None
    return load_checkpoint(str(CHECKPOINT_PATH), num_classes=NUM_CLASSES, device="cpu")


def preprocess(image: Image.Image) -> torch.Tensor:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.unsqueeze(0), resized


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, rgb in enumerate(PALETTE):
        color[mask == class_idx] = rgb
    return color


def overlay(image: Image.Image, color_mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    base = np.array(image).astype(np.float32)
    blended = (1 - alpha) * base + alpha * color_mask.astype(np.float32)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def main():
    st.set_page_config(page_title="Facade Energy-Relevance Analyzer", layout="wide")
    st.title("Building Facade Energy-Relevance Analyzer")
    st.caption(
        "Upload a building facade photo to estimate window-to-wall ratio (WWR), "
        "wall tone (solar absorptance proxy), and shading coverage from a "
        "fine-tuned DeepLabV3 segmentation model."
    )

    model = get_model()
    if model is None:
        st.warning(
            f"No trained model checkpoint found at `{CHECKPOINT_PATH}`. "
            "Train one with `notebooks/train_segmentation.ipynb` (recommended: "
            "Google Colab) or `python src/train.py`, then re-run this app."
        )
        return

    uploaded = st.file_uploader("Upload a facade photo", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a JPG/PNG photo of a building facade to get started.")
        return

    image = Image.open(uploaded).convert("RGB")
    input_tensor, resized_image = preprocess(image)

    with torch.no_grad():
        logits = model(input_tensor)["out"]
    pred_mask = logits.argmax(dim=1).squeeze(0).numpy()

    color_mask = mask_to_color(pred_mask)
    overlay_image = overlay(resized_image, color_mask)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(resized_image, use_container_width=True)
    with col2:
        st.subheader("Segmentation overlay")
        st.image(overlay_image, use_container_width=True)

    st.subheader("Legend")
    legend_cols = st.columns(len(CLASS_NAMES))
    for col, name, rgb in zip(legend_cols, CLASS_NAMES, PALETTE):
        with col:
            swatch = Image.new("RGB", (40, 20), rgb)
            st.image(swatch)
            st.caption(name)

    metrics = compute_all_metrics(np.array(resized_image), pred_mask)

    st.subheader("Estimated metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Window-to-Wall Ratio", f"{metrics['wwr']:.1%}")
    wall_tone = metrics["wall_tone"]
    m2.metric(
        "Wall tone (0=dark, 255=light)",
        f"{wall_tone:.0f}" if wall_tone is not None else "n/a (no wall detected)",
    )
    m3.metric("Shading coverage (near windows)", f"{metrics['shading_coverage']:.1%}")

    st.caption(
        "These are visually-derived proxies, not a full energy audit — see the "
        "project README for explicit scope boundaries (orientation, insulation, "
        "glazing type, etc. are out of scope for a single photo)."
    )


if __name__ == "__main__":
    main()
