"""Model factory."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from vocseg.models.deeplabv3plus import DeepLabV3Plus
from vocseg.models.sam2_adapter import SAM2SemanticSegmentor
from vocseg.models.segformer import SegFormer
from vocseg.models.unet import UNetResNet


def build_model(model_cfg: dict[str, Any], num_classes: int) -> nn.Module:
    family = model_cfg["family"]
    if family == "unet":
        return UNetResNet(
            num_classes=num_classes,
            backbone=model_cfg.get("backbone", "resnet34"),
            pretrained=model_cfg.get("pretrained", True),
        )
    if family == "deeplabv3plus":
        return DeepLabV3Plus(
            num_classes=num_classes,
            backbone=model_cfg.get("backbone", "resnet50"),
            pretrained=model_cfg.get("pretrained", True),
        )
    if family == "segformer":
        return SegFormer(
            num_classes=num_classes,
            backbone_name=model_cfg.get("backbone", "segformer_b2"),
            pretrained=model_cfg.get("pretrained", True),
            embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        )
    if family == "sam2_semantic":
        return SAM2SemanticSegmentor(
            num_classes=num_classes,
            sam2_cfg=model_cfg["sam2"],
        )
    raise ValueError(f"Unknown model family: {family}")
