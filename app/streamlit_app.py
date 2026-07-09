"""Streamlit demo: upload a building facade photo, get a segmentation overlay
plus computed WWR / wall-tone / shading-coverage metrics, and (if the
detector checkpoint is trained) window/door counts with an average
window-size estimate.

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, PALETTE  # noqa: E402
from detect_model import CLASS_NAMES as DETECT_CLASS_NAMES, count_detections, load_detector  # noqa: E402
from metrics import compute_all_metrics, compute_avg_window_share  # noqa: E402
from model import load_checkpoint  # noqa: E402

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pt"
DETECTOR_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "best_detector.pt"
IMAGE_SIZE = 384
DETECT_CONF = 0.25
DETECT_BOX_COLORS = {"window": (42, 120, 214), "door": (230, 73, 72), "building": (237, 161, 0), "gate": (74, 58, 167)}


@st.cache_resource
def get_model():
    if not CHECKPOINT_PATH.exists():
        return None
    return load_checkpoint(str(CHECKPOINT_PATH), num_classes=NUM_CLASSES, device="cpu")


@st.cache_resource
def get_detector():
    if not DETECTOR_CHECKPOINT_PATH.exists():
        return None
    return load_detector(str(DETECTOR_CHECKPOINT_PATH))


def draw_detections(image: Image.Image, results) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    boxes = results.boxes
    if boxes is None:
        return annotated
    for xyxy, cls_idx, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
        if conf < DETECT_CONF:
            continue
        name = DETECT_CLASS_NAMES[int(cls_idx)]
        color = DETECT_BOX_COLORS.get(name, (255, 255, 255))
        draw.rectangle(xyxy, outline=color, width=3)
        draw.text((xyxy[0] + 3, max(xyxy[1] - 14, 0)), name, fill=color)
    return annotated


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
    detector = get_detector()
    if model is None:
        st.warning(
            f"No trained segmentation checkpoint found at `{CHECKPOINT_PATH}`. "
            "Train one with `notebooks/train_segmentation.ipynb` (recommended: "
            "Google Colab) or `python src/train.py`, then re-run this app."
        )
        return
    if detector is None:
        st.info(
            f"No trained window/door detector found at `{DETECTOR_CHECKPOINT_PATH}` — "
            "counts won't be shown. Train one with `notebooks/train_detection.ipynb` "
            "or `python src/detect_train.py`."
        )

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

    counts = None
    detection_image = None
    if detector is not None:
        # Run on the original image, not the square-cropped `resized_image` --
        # that resize forces a fixed aspect ratio for the segmentation model,
        # which distorts objects and measurably hurts detection (verified:
        # dropped a real detection to zero on a test image). YOLO handles
        # arbitrary aspect ratios natively, so it doesn't need that resize.
        results = detector.predict(image, conf=DETECT_CONF, verbose=False)[0]
        counts = count_detections(results, conf=DETECT_CONF)
        detection_image = draw_detections(image, results)

    if detection_image is not None:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)
        col3 = None
    with col1:
        st.subheader("Original")
        st.image(resized_image, use_container_width=True)
    with col2:
        st.subheader("Segmentation overlay")
        st.image(overlay_image, use_container_width=True)
    if col3 is not None:
        with col3:
            st.subheader("Detected windows/doors")
            st.image(detection_image, use_container_width=True)

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

    if counts is not None:
        st.subheader("Window & door counts")
        c1, c2, c3 = st.columns(3)
        c1.metric("Windows detected", counts["window"])
        c2.metric("Doors detected", counts["door"])
        avg_share = compute_avg_window_share(metrics["wwr"], counts["window"])
        c3.metric(
            "Avg. window size (share of facade)",
            f"{avg_share:.1%}" if avg_share is not None else "n/a (no windows detected)",
        )
        st.caption(
            "Counts come from a separate object-detection model (not the segmentation "
            "model above). Note: some multi-pane windows or glass doors may be counted "
            "as several boxes rather than one, so counts can run high on glass-heavy "
            "facades — see the README for details."
        )

    st.caption(
        "These are visually-derived proxies, not a full energy audit — see the "
        "project README for explicit scope boundaries (orientation, insulation, "
        "glazing type, etc. are out of scope for a single photo)."
    )


if __name__ == "__main__":
    main()
