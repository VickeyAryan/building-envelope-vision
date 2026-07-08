# Building Facade Energy-Relevance Analyzer

A computer vision project that analyzes building facade photos to estimate visually-derivable indicators relevant to building energy performance — primarily **window-to-wall ratio (WWR)**, alongside supporting metrics like wall tone (solar absorptance proxy) and shading coverage — using semantic segmentation.

## Problem Statement

Window-to-wall ratio is a well-established driver of building energy performance: it strongly influences solar heat gain, daylighting, and heating/cooling loads. Manually measuring WWR and related facade characteristics across many buildings (e.g., for urban-scale energy modeling) is slow and labor-intensive. This project explores whether a pretrained segmentation model, fine-tuned on a facade dataset, can automatically extract these metrics from a single photo.

This builds on the same core technique (transfer learning, PyTorch) as the Fabric Defect Detector project, applied to a **segmentation** task instead of classification — extracting pixel-level facade elements rather than a single whole-image label.

## What This Project Measures (and What It Doesn't)

**Derived from the image (in scope):**
- **Window-to-Wall Ratio (WWR)** — window pixel area ÷ total facade pixel area
- **Wall tone / solar absorptance proxy** — average brightness of wall-class pixels (darker walls absorb more heat)
- **Shading coverage proxy** — balcony/cornice pixel area near windows, as a rough indicator of passive shading

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

## Approach

- **Task type**: semantic segmentation (pixel-level classification), not whole-image classification
- **Model**: fine-tune a pretrained **DeepLabV3** (ResNet backbone, built into `torchvision`) on the CMP facade labels
- **Training environment**: Google Colab (free GPU); local machine for data prep, inference, and the demo app
- **Framework**: PyTorch + torchvision

## Project Structure

```
building-envelope-vision/
├── data/                     # downloaded CMP dataset (gitignored)
├── notebooks/
│   └── train_segmentation.ipynb   # Colab training notebook
├── src/
│   ├── data_loader.py        # dataset loading & mask preprocessing
│   ├── model.py               # DeepLabV3 setup (pretrained backbone + fine-tuned head)
│   ├── train.py               # training loop
│   ├── evaluate.py            # IoU, per-class segmentation metrics
│   └── metrics.py             # WWR, wall-tone, shading-proxy calculations from predicted masks
├── app/
│   └── streamlit_app.py       # upload facade photo -> visual mask overlay + computed metrics
├── models/
│   └── best_model.pt
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

**Run the demo app locally:**
```bash
streamlit run app/streamlit_app.py
```
Upload a building facade photo → view the segmented overlay (window/wall/shading regions highlighted) plus computed WWR, wall-tone, and shading-proxy values.

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
- Explore YOLO-based bounding-box detection as a lighter-weight alternative, trading precision for speed

## Tech Stack

Python · PyTorch · torchvision (DeepLabV3) · Hugging Face `datasets` · Streamlit

## License

MIT
