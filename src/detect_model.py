"""YOLO object-detection setup for counting individual windows and doors.

Unlike the segmentation model (src/model.py), this is a thin wrapper around
ultralytics' YOLO class rather than a hand-rolled architecture: ultralytics
owns model construction, training loop, and checkpoint format internally.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

CLASS_NAMES = ["building", "door", "gate", "window"]


def resolve_data_yaml(yaml_path: str) -> str:
    """Rewrites `path:` in a YOLO data.yaml to an absolute path before handing
    it to ultralytics.

    Ultralytics resolves a relative `path:` against the current working
    directory (or its own datasets-dir setting), not against the yaml file's
    own location -- so a relative `path: .` breaks unless the script happens
    to be run from that exact directory. Rewriting it to an absolute path at
    runtime keeps the committed yaml portable across machines/clone
    locations, since nothing machine-specific ends up in git.
    """
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    config["path"] = str(yaml_path.parent)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(config, tmp)
    tmp.close()
    return tmp.name


def build_detector(pretrained: bool = True) -> YOLO:
    """Builds a YOLOv8-nano detector. `pretrained` loads COCO weights as the
    starting point for fine-tuning; nano is chosen so local CPU smoke tests
    stay feasible (segmentation's ResNet50 backbone would be far slower here).
    """
    return YOLO("yolov8n.pt" if pretrained else "yolov8n.yaml")


def load_detector(checkpoint_path: str) -> YOLO:
    """Loads a fine-tuned detector from a checkpoint saved by src/detect_train.py."""
    return YOLO(checkpoint_path)


def count_detections(results, class_names: list[str] = CLASS_NAMES, conf: float = 0.25) -> dict:
    """Turns one ultralytics `Results` object into per-class counts.

    `results` is a single element from the list returned by `model.predict(...)`.
    Boxes below `conf` are already filtered out if `predict` was called with
    the same `conf` threshold; this also re-filters defensively in case it wasn't.
    """
    counts = {name: 0 for name in class_names}
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return counts
    for cls_idx, box_conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        if box_conf < conf:
            continue
        name = class_names[int(cls_idx)]
        counts[name] = counts.get(name, 0) + 1
    return counts
