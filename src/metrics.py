"""Domain metrics derived from a predicted facade segmentation mask.

All functions take a class-index mask (H, W) of ints in [0, NUM_CLASSES) and,
where relevant, the source RGB image, and return plain Python floats so they
are easy to display in the demo app or log during evaluation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from data_loader import CLASS_NAMES

BACKGROUND = CLASS_NAMES.index("background")
FACADE = CLASS_NAMES.index("facade")
WINDOW = CLASS_NAMES.index("window")
BALCONY = CLASS_NAMES.index("balcony")
CORNICE = CLASS_NAMES.index("cornice")

# Everything that isn't background counts as part of the building envelope.
BUILDING_CLASSES = [i for i in range(len(CLASS_NAMES)) if i != BACKGROUND]


def compute_wwr(mask: np.ndarray) -> float:
    """Window-to-Wall Ratio = window pixels / total building-facade pixels.

    "Total building-facade pixels" is every non-background class (wall,
    window, door, sill, etc.) rather than just the plain-wall class, since
    WWR is conventionally window area over total exterior envelope area.
    Returns 0.0 if no building pixels are detected at all.
    """
    building_area = np.isin(mask, BUILDING_CLASSES).sum()
    if building_area == 0:
        return 0.0
    window_area = (mask == WINDOW).sum()
    return float(window_area / building_area)


def compute_wall_tone(image_rgb: np.ndarray, mask: np.ndarray) -> float | None:
    """Average brightness (0-255) of pixels classified as plain wall ("facade").

    Darker walls absorb more solar heat (lower solar reflectance), so this
    value is a rough proxy for solar absorptance. Returns None if no wall
    pixels were predicted (e.g., a tightly cropped window photo).
    """
    wall_pixels = mask == FACADE
    if wall_pixels.sum() == 0:
        return None
    gray = image_rgb.astype(np.float32).mean(axis=-1)
    return float(gray[wall_pixels].mean())


def compute_shading_coverage(mask: np.ndarray, dilation_px: int = 15) -> float:
    """Fraction of window area that sits adjacent to a balcony or cornice.

    Dilates the window mask by `dilation_px` and measures overlap with
    balcony/cornice pixels, as a rough proxy for passive shading coverage.
    Returns 0.0 if there are no windows.
    """
    window_mask = mask == WINDOW
    window_area = window_mask.sum()
    if window_area == 0:
        return 0.0

    shading_mask = np.isin(mask, [BALCONY, CORNICE])
    if shading_mask.sum() == 0:
        return 0.0

    dilated_window = binary_dilation(window_mask, iterations=dilation_px)
    shaded_window_area = (dilated_window & shading_mask).sum()
    return float(min(shaded_window_area / window_area, 1.0))


def compute_all_metrics(image_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Bundles all facade metrics for a single image/mask pair."""
    return {
        "wwr": compute_wwr(mask),
        "wall_tone": compute_wall_tone(image_rgb, mask),
        "shading_coverage": compute_shading_coverage(mask),
    }
