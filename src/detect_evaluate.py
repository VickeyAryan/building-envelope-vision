"""Evaluates a fine-tuned window/door detector: wraps ultralytics' built-in
`model.val()`, which already computes mAP50/mAP50-95/precision/recall per
class -- no need to hand-roll a confusion matrix the way src/evaluate.py
does for the segmentation model.

    python src/detect_evaluate.py --checkpoint models/best_detector.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from detect_model import CLASS_NAMES, load_detector, resolve_data_yaml

DEFAULT_DATA_YAML = str(
    Path(__file__).resolve().parent.parent / "data" / "window_door_detection" / "data.yaml"
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned window/door detector.")
    parser.add_argument("--checkpoint", default="models/best_detector.pt")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()

    model = load_detector(args.checkpoint)
    results = model.val(data=resolve_data_yaml(args.data), imgsz=args.imgsz, split=args.split)

    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print("Per-class mAP50:")
    for idx, ap50 in enumerate(results.box.ap50):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
        print(f"  {name:12s}: {ap50:.4f}")


if __name__ == "__main__":
    main()
