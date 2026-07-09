"""Fine-tunes YOLOv8 on the window/door/building/gate detection dataset.

Unlike src/train.py, there's no custom training loop here -- ultralytics'
`model.train(...)` owns batching, augmentation, checkpointing, and the epoch
loop internally. This script just configures that call and copies the
resulting best checkpoint to a stable, predictable path.

    python src/detect_train.py --epochs 50 --imgsz 416 --batch 8
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from detect_model import build_detector, resolve_data_yaml

DEFAULT_DATA_YAML = str(
    Path(__file__).resolve().parent.parent / "data" / "window_door_detection" / "data.yaml"
)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the window/door detection dataset.")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 'cpu' or '0' for first GPU; ultralytics auto-picks if unset")
    parser.add_argument("--output", default="models/best_detector.pt")
    parser.add_argument("--run-name", default="facade_detector")
    args = parser.parse_args()

    model = build_detector(pretrained=True)
    train_kwargs = dict(
        data=resolve_data_yaml(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.run_name,
        exist_ok=True,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    # Ask ultralytics where it actually saved things, rather than guessing the
    # path ourselves (its default project/name-to-directory logic isn't
    # simple string concatenation of what we pass in).
    best_weights = model.trainer.save_dir / "weights" / "best.pt"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, output_path)
    print(f"Best checkpoint copied to {output_path}")


if __name__ == "__main__":
    main()
