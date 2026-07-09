# Building Facade Energy-Relevance Analyzer

A computer vision project that analyzes building facade photos to estimate visually-derivable indicators relevant to building energy performance — primarily **window-to-wall ratio (WWR)**, alongside supporting metrics like wall tone (solar absorptance proxy) and shading coverage — using semantic segmentation. A second model adds **window/door counting** via object detection.

## Problem Statement

Window-to-wall ratio is a well-established driver of building energy performance: it strongly influences solar heat gain, daylighting, and heating/cooling loads. Manually measuring WWR and related facade characteristics across many buildings (e.g., for urban-scale energy modeling) is slow and labor-intensive. This project explores whether a pretrained segmentation model, fine-tuned on a facade dataset, can automatically extract these metrics from a single photo.

This builds on the same core technique (transfer learning, PyTorch) as the Fabric Defect Detector project, applied to a **segmentation** task instead of classification — extracting pixel-level facade elements rather than a single whole-image label.

## What This Project Measures (and What It Doesn't)

**Derived from the image (in scope):**
- **Window-to-Wall Ratio (WWR)** — window pixel area ÷ total facade pixel area
- **Wall tone / solar absorptance proxy** — average brightness of wall-class pixels (darker walls absorb more heat)
- **Shading coverage proxy** — balcony/cornice pixel area near windows, as a rough indicator of passive shading
- **Window & door counts** — how many distinct windows/doors are visible, via a separate object-detection model (see [Window & Door Counting](#window--door-counting) below)
- **Average window size** — WWR ÷ window count, a relative (not absolute) indicator of whether a facade has many small windows or few large ones

**Not derivable from a single photo (explicitly out of scope):**
- Building orientation (needs GPS/compass data or multiple facade views)
- Insulation quality or wall material composition (invisible from outside; needs thermal imaging or building records)
- Glazing type (single vs. double-pane)
- HVAC systems, occupancy, climate zone

This scope boundary is intentional — the project demonstrates what computer vision *can* contribute to a building energy assessment, not a full energy audit replacement.

## Dataset

**CMP Facade Database**
- 606 rectified, manually annotated building facade images
- Pixel-level segmentation labels across 12 classes: facade, window, door, sill, cornice, pillar, balcony, blind, shop, molding, deco, background
- Source: Center for Machine Perception (CMP), various cities and architectural styles
- Access via Hugging Face:
```python
from datasets import load_dataset
dataset = load_dataset("Xpitfire/cmp_facade")
```
- Also available on Kaggle (`adlteam/facade-dataset`) as an alternative source

**Window/Door Detection Dataset** (for counting — see [Window & Door Counting](#window--door-counting))
- 324 images (276 train / 48 valid), bounding-box labels for 4 classes: building, door, gate, window
- Source: ["building, door, gate, window"](https://universe.roboflow.com/facade-features/building-door-gate-window) by Facade Features, Roboflow Universe, CC BY 4.0 (see `data/window_door_detection/ATTRIBUTION.md`)
- Vendored directly in this repo (Git LFS) under `data/window_door_detection/`, unlike the CMP dataset — Roboflow requires a personal account/API key to download, so committing the files keeps `git clone` self-contained for everyone

## Approach

- **Task type**: semantic segmentation (pixel-level classification) for WWR/wall-tone/shading, plus object detection (bounding boxes) for window/door counting — two different tasks needing two different model types, not one model doing both
- **Segmentation model**: fine-tune a pretrained **DeepLabV3** (ResNet backbone, built into `torchvision`) on the CMP facade labels
- **Detection model**: fine-tune a pretrained **YOLOv8-nano** (via `ultralytics`) on the window/door detection dataset
- **Training environment**: Google Colab (free GPU); local machine for data prep, inference, and the demo app
- **Framework**: PyTorch + torchvision (segmentation), Ultralytics YOLO (detection)

## Window & Door Counting

Semantic segmentation labels *pixels*, not *objects* — it can tell you what fraction of a facade is window, but not how many separate windows there are (a row of 10 windows and one giant window can look identical in pixel-area terms). Counting individual objects needs a different technique: object detection, which draws one box per object.

We initially tried deriving counts from the segmentation model's own masks (grouping connected "window" pixels into blobs), but testing this on real CMP ground-truth masks produced nonsense — e.g. 73 separate "door" blobs in one photo, because CMP's labels are broad semantic shapes, not per-object outlines. So counting uses a second, independently-trained model (YOLOv8) on a purpose-built bounding-box dataset instead.

**Known limitation**: the detection dataset sometimes annotates one multi-pane window or glass door as several sub-boxes (one per pane) rather than one box per window a person would count. This means counts can run high on glass-heavy/modern facades — a real limitation of the source annotations, not the model.

## Project Structure

```
building-envelope-vision/
├── data/
│   └── window_door_detection/     # vendored detection dataset (Git LFS) + ATTRIBUTION.md
│                                   # (CMP segmentation dataset is downloaded fresh via Hugging Face, not stored here)
├── notebooks/
│   ├── train_segmentation.ipynb   # Colab training notebook (segmentation)
│   └── train_detection.ipynb      # Colab training notebook (window/door counting)
├── src/
│   ├── data_loader.py        # CMP dataset loading & mask preprocessing
│   ├── model.py               # DeepLabV3 setup (pretrained backbone + fine-tuned head)
│   ├── train.py               # segmentation training loop
│   ├── evaluate.py            # IoU, per-class segmentation metrics
│   ├── metrics.py             # WWR, wall-tone, shading-proxy, avg-window-size calculations
│   ├── detect_model.py        # YOLOv8 setup (wraps ultralytics.YOLO)
│   ├── detect_train.py        # detection training CLI
│   └── detect_evaluate.py     # mAP evaluation CLI
├── app/
│   └── streamlit_app.py       # upload facade photo -> segmentation overlay + detection boxes + metrics
├── models/
│   ├── best_model.pt          # segmentation checkpoint
│   └── best_detector.pt       # detection checkpoint
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/VickeyAryan/building-envelope-vision.git
cd building-envelope-vision

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

Download the dataset:
```python
from datasets import load_dataset
dataset = load_dataset("Xpitfire/cmp_facade")
```

## Usage

**Train the model** (recommended: run `notebooks/train_segmentation.ipynb` on Google Colab for a GPU)
```bash
python src/train.py --epochs 20 --batch-size 8 --image-size 384
```
CPU training is slow; `--max-train-samples` / `--max-val-samples` cap the dataset size for quick local smoke tests, e.g. `python src/train.py --epochs 5 --max-train-samples 150 --max-val-samples 40 --image-size 224`.

**Evaluate a checkpoint on the held-out test split:**
```bash
python src/evaluate.py --checkpoint models/best_model.pt
```

**Train the window/door counter** (recommended: run `notebooks/train_detection.ipynb` on Google Colab for a GPU)
```bash
python src/detect_train.py --epochs 50 --imgsz 416 --batch 8
```
YOLOv8-nano is far lighter than the segmentation model, so even CPU-only local runs are reasonably fast.

**Evaluate the detector:**
```bash
python src/detect_evaluate.py --checkpoint models/best_detector.pt
```

**Run the demo app locally:**
```bash
streamlit run app/streamlit_app.py
```
Upload a building facade photo → view the segmented overlay (window/wall/shading regions highlighted), detected window/door boxes with counts, and computed WWR, wall-tone, shading-proxy, and average-window-size values. The app works with just the segmentation checkpoint present — the detector/counting panel only appears once `models/best_detector.pt` exists.

## Results

Full training run on Google Colab (T4 GPU): 606 images (378 train / 114 val / 114 test), 384×384, 60 epochs, batch size 8, Adam lr 1e-4, using `notebooks/train_segmentation.ipynb`.

Evaluated on the full 114-image held-out test split (`python src/evaluate.py --image-size 384`):

| Metric | Value |
|---|---|
| Pixel accuracy | 0.717 |
| Mean IoU | 0.438 |

| Class | IoU |
|---|---|
| facade | 0.702 |
| door | 0.623 |
| window | 0.589 |
| pillar | 0.497 |
| sill | 0.441 |
| blind | 0.426 |
| cornice | 0.378 |
| balcony | 0.361 |
| shop | 0.343 |
| deco | 0.268 |
| molding | 0.188 |
| background | n/a (no background pixels in the test split) |

Facade, door, and window — the classes WWR depends on most — segment well. Small/rare decorative classes (molding, deco) are still the weakest, as expected given how little facade area they typically cover.

Earlier local CPU-only smoke run (kept for reference, not representative of the model's real capability): 150 training images, 224×224, ~1 epoch — pixel accuracy 0.475, mean IoU 0.157. The jump to full-dataset/full-resolution/60-epoch GPU training roughly tripled mean IoU, confirming the extra training budget mattered far more than any pipeline issue.

**Window/door detection (counting):** 30-epoch run on Google Colab (T4 GPU), 416×416, batch 16, via `notebooks/train_detection.ipynb`.

Evaluated on the 48-image validation split (`python src/detect_evaluate.py --imgsz 416`):

| Metric | Value |
|---|---|
| mAP50 | 0.663 |
| mAP50-95 | 0.443 |
| Precision | 0.744 |
| Recall | 0.598 |

| Class | mAP50 |
|---|---|
| building | 0.856 |
| gate | 0.669 |
| window | 0.615 |
| door | 0.510 |

On a real validation photo, the model now correctly counts every object (1 building, 2 doors, 6 windows — an exact match to ground truth), versus the earlier 2-epoch smoke test which found only 1 window and 0 doors on the same image. Gate has the fewest training examples (69 boxes total) so its mAP is noisier than the others despite the relatively high number.

Earlier local CPU-only smoke run (kept for reference): 2 epochs, 320×320 — mAP50 0.335, mAP50-95 0.195, recall 0.279. The jump to 30 epochs on a GPU roughly doubled mAP50 and more than doubled recall, the same pattern seen with the segmentation model's full training run mattering more than any pipeline change.

`src/detect_train.py` doesn't currently support `--resume`; a longer run (e.g. 100 epochs, the notebook's default) could push these numbers further.

`src/train.py --resume <checkpoint>` can continue training from any saved checkpoint if you want to push these numbers further.

## What This Project Demonstrates

- Semantic segmentation via transfer learning (pretrained DeepLabV3 fine-tuning)
- Deriving domain-meaningful metrics (WWR, solar absorptance proxy) from raw pixel predictions
- End-to-end pipeline: data loading → training → evaluation → deployment (demo app)
- Explicit, honest scoping of what vision-based analysis can and cannot tell you about building energy performance

## Future Improvements

- Multi-photo input to estimate building orientation and combine multiple facades
- Compare against ground-truth WWR values (if available) to validate accuracy
- Extend shading-proxy metric to distinguish shading device *type* (overhang vs. balcony vs. vegetation)
- Longer detector training (100+ epochs) to push mAP further, and `--resume` support in `detect_train.py` for chaining runs
- Post-process detections to merge multi-pane window/glass-door sub-boxes into one count per human-intuitive "window," fixing the known overcounting caveat

## Tech Stack

Python · PyTorch · torchvision (DeepLabV3) · Ultralytics YOLOv8 · Hugging Face `datasets` · Roboflow · Streamlit

## License

MIT
