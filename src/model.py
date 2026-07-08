"""DeepLabV3 (ResNet-50 backbone) setup for facade semantic segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Builds a DeepLabV3-ResNet50 with its classifier heads replaced for
    `num_classes` facade classes. The ResNet backbone keeps its COCO/ImageNet
    pretrained weights; only the segmentation heads are freshly initialized
    and meant to be fine-tuned.
    """
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    # aux_loss=True always, regardless of `pretrained`, so the model
    # architecture (and therefore its state_dict keys) is identical whether
    # building fresh for training or rebuilding to load a fine-tuned checkpoint.
    model = deeplabv3_resnet50(weights=weights, weights_backbone=weights_backbone, aux_loss=True)

    in_channels = model.classifier[0].convs[0][0].in_channels
    model.classifier = DeepLabHead(in_channels, num_classes)

    aux_in_channels = model.aux_classifier[0].in_channels
    model.aux_classifier = FCNHead(aux_in_channels, num_classes)

    return model


def load_checkpoint(checkpoint_path: str, num_classes: int, device: str = "cpu") -> nn.Module:
    """Loads a fine-tuned model from a checkpoint saved by src/train.py."""
    model = build_model(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
