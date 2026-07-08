"""Fine-tunes DeepLabV3 on the CMP Facade Database.

Recommended to run on Google Colab (free GPU) via
notebooks/train_segmentation.ipynb, which imports this module. Can also run
locally on CPU for smaller experiments (slow, but works):

    python src/train.py --epochs 20 --batch-size 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data_loader import NUM_CLASSES, get_datasets
from evaluate import evaluate
from model import build_model


def train_one_epoch(model, dataloader, optimizer, criterion, device: str) -> float:
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs["out"], masks)
        if "aux" in outputs:
            loss = loss + 0.4 * criterion(outputs["aux"], masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 on the CMP Facade Database.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="models/best_model.pt")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap the training set size, for quick local CPU smoke tests (full dataset if unset).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Cap the validation set size, for quick local CPU smoke tests (full dataset if unset).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing checkpoint to continue training from (e.g. after a "
        "previous run was interrupted). Useful for chaining short CPU runs.",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print("Loading CMP Facade Database...")
    train_ds, val_ds, _ = get_datasets(image_size=args.image_size)
    if args.max_train_samples is not None:
        train_ds = Subset(train_ds, list(range(min(args.max_train_samples, len(train_ds)))))
    if args.max_val_samples is not None:
        val_ds = Subset(val_ds, list(range(min(args.max_val_samples, len(val_ds)))))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    model = build_model(num_classes=NUM_CLASSES, pretrained=(args.resume is None)).to(args.device)
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_miou = -1.0
    if args.resume is not None:
        best_miou = evaluate(model, val_loader, args.device)["mean_iou"]
        print(f"Resumed model's current val mIoU: {best_miou:.4f}")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_results = evaluate(model, val_loader, args.device)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_mIoU={val_results['mean_iou']:.4f} "
            f"| val_pixel_acc={val_results['pixel_accuracy']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if val_results["mean_iou"] > best_miou:
            best_miou = val_results["mean_iou"]
            torch.save(model.state_dict(), output_path)
            print(f"  -> new best (mIoU={best_miou:.4f}), saved to {output_path}")

    print(f"Training complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
