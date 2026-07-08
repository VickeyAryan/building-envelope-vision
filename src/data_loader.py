"""Dataset loading and mask preprocessing for the CMP Facade Database.

The CMP dataset (`Xpitfire/cmp_facade` on Hugging Face) ships each example as an
RGB photo (`pixel_values`) plus a palette-indexed PNG mask (`label`) whose pixel
values are already class indices 0-11. The class order below matches the
dataset's official palette (verified against the dataset card + decoded PNG
palette bytes), not just the prose order in the card.
"""
from __future__ import annotations

import io

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

CLASS_NAMES = [
    "background",
    "facade",
    "window",
    "door",
    "cornice",
    "sill",
    "balcony",
    "blind",
    "deco",
    "molding",
    "pillar",
    "shop",
]
NUM_CLASSES = len(CLASS_NAMES)

# RGB color for each class index, taken directly from the dataset's own PNG
# palette (decoded from a sample mask) so overlays match the source labels.
PALETTE = [
    (0, 0, 0),        # background
    (0, 0, 170),      # facade
    (0, 0, 255),      # window
    (0, 85, 255),     # door
    (0, 170, 255),    # cornice
    (0, 255, 255),    # sill
    (85, 255, 170),   # balcony
    (170, 255, 85),   # blind
    (255, 255, 0),    # deco
    (255, 170, 0),    # molding
    (255, 85, 0),     # pillar
    (255, 0, 0),      # shop
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_cmp_facade(hf_name: str = "Xpitfire/cmp_facade"):
    """Returns the raw HF DatasetDict with 'train', 'test', 'eval' splits."""
    return load_dataset(hf_name)


def _to_pil(value) -> Image.Image:
    """The dataset stores `pixel_values`/`label` as raw {bytes, path} structs
    rather than a decoded HF `Image` feature, so both fields need manual
    PNG decoding via Pillow.
    """
    if isinstance(value, Image.Image):
        return value
    return Image.open(io.BytesIO(value["bytes"]))


class FacadeSegDataset(Dataset):
    """Wraps one split of the CMP facade HF dataset as a torch Dataset.

    Images are resized (bilinear) and masks resized (nearest, to avoid
    inventing intermediate class labels at edges) to a fixed square size so
    examples of varying aspect ratio can be batched together.
    """

    def __init__(self, hf_split, image_size: int = 384, augment: bool = False):
        self.hf_split = hf_split
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int):
        example = self.hf_split[idx]
        image = _to_pil(example["pixel_values"]).convert("RGB")
        # Mask is a palette-indexed ("P" mode) PNG whose pixel values ARE the
        # class indices. Do NOT convert to "L"/"RGB" -- that would replace
        # indices with palette-derived luminance/color values instead of
        # preserving the class labels.
        mask = _to_pil(example["label"])
        if mask.mode != "P":
            mask = mask.convert("P")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment and np.random.rand() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask_arr = np.array(mask, dtype=np.int64)
        mask_arr = np.clip(mask_arr, 0, NUM_CLASSES - 1)
        mask_t = torch.from_numpy(mask_arr)

        return image_t, mask_t


def get_datasets(image_size: int = 384, hf_name: str = "Xpitfire/cmp_facade"):
    """Convenience helper: returns (train_ds, val_ds, test_ds) torch Datasets."""
    raw = load_cmp_facade(hf_name)
    train_ds = FacadeSegDataset(raw["train"], image_size=image_size, augment=True)
    val_ds = FacadeSegDataset(raw["eval"], image_size=image_size, augment=False)
    test_ds = FacadeSegDataset(raw["test"], image_size=image_size, augment=False)
    return train_ds, val_ds, test_ds


def denormalize(image_t: torch.Tensor) -> torch.Tensor:
    """Undo the ImageNet normalization for display purposes. Returns a [0,1] tensor."""
    mean = torch.tensor(IMAGENET_MEAN, device=image_t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_t.device).view(3, 1, 1)
    return (image_t * std + mean).clamp(0, 1)
