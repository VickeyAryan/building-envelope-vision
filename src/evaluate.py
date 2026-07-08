"""Segmentation evaluation: confusion matrix, per-class and mean IoU."""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import CLASS_NAMES, NUM_CLASSES, get_datasets
from model import load_checkpoint


@torch.no_grad()
def compute_confusion_matrix(model, dataloader, device: str, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Accumulates a (num_classes, num_classes) confusion matrix over a dataloader.

    Rows are ground truth, columns are predictions.
    """
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    for images, masks in dataloader:
        images = images.to(device)
        logits = model(images)["out"]
        preds = logits.argmax(dim=1).cpu().numpy()
        gts = masks.numpy()

        valid = (gts >= 0) & (gts < num_classes)
        idx = num_classes * gts[valid].astype(np.int64) + preds[valid].astype(np.int64)
        conf_mat += np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf_mat


def iou_from_confusion_matrix(conf_mat: np.ndarray) -> np.ndarray:
    """Per-class IoU = TP / (TP + FP + FN). NaN for classes absent from both gt and preds."""
    tp = np.diag(conf_mat).astype(np.float64)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    return iou


def evaluate(model, dataloader, device: str) -> dict:
    conf_mat = compute_confusion_matrix(model, dataloader, device)
    iou = iou_from_confusion_matrix(conf_mat)
    per_class_iou = {name: (float(v) if not np.isnan(v) else None) for name, v in zip(CLASS_NAMES, iou)}
    mean_iou = float(np.nanmean(iou))
    pixel_acc = float(np.diag(conf_mat).sum() / conf_mat.sum())
    return {
        "confusion_matrix": conf_mat,
        "per_class_iou": per_class_iou,
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned facade segmentation model.")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _, _, test_ds = get_datasets(image_size=args.image_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = load_checkpoint(args.checkpoint, num_classes=NUM_CLASSES, device=args.device)
    results = evaluate(model, test_loader, args.device)

    print(f"Pixel accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print("Per-class IoU:")
    for name, val in results["per_class_iou"].items():
        print(f"  {name:12s}: {'n/a' if val is None else f'{val:.4f}'}")


if __name__ == "__main__":
    main()
