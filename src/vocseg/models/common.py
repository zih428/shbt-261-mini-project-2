"""Shared building blocks for models."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.models import resnet18, resnet34, resnet50


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int | None = None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


RESNET_BUILDERS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
}


def build_resnet_encoder(
    backbone: str,
    pretrained: bool,
    replace_stride_with_dilation: tuple[bool, bool, bool] | None = None,
) -> nn.Module:
    if backbone not in RESNET_BUILDERS:
        raise ValueError(f"Unsupported ResNet backbone: {backbone}")
    builder, weights = RESNET_BUILDERS[backbone]
    kwargs = {}
    if replace_stride_with_dilation is not None:
        kwargs["replace_stride_with_dilation"] = replace_stride_with_dilation
    if not pretrained:
        return builder(weights=None, **kwargs)

    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if cache_path.exists():
        model = builder(weights=None, **kwargs)
        state_dict = torch.load(cache_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model

    try:
        return builder(weights=weights, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load pretrained weights for {backbone}. "
            f"Cache the checkpoint first with `uv run python scripts/fetch_pretrained_assets.py --resnets {backbone}`."
        ) from exc


def extract_resnet_feature_pyramid(backbone: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    x0 = backbone.relu(backbone.bn1(backbone.conv1(x)))
    x1 = backbone.layer1(backbone.maxpool(x0))
    x2 = backbone.layer2(x1)
    x3 = backbone.layer3(x2)
    x4 = backbone.layer4(x3)
    return [x0, x1, x2, x3, x4]


def infer_resnet_channels(backbone: str) -> list[int]:
    if backbone in {"resnet18", "resnet34"}:
        return [64, 64, 128, 256, 512]
    if backbone == "resnet50":
        return [64, 256, 512, 1024, 2048]
    raise ValueError(f"Unsupported ResNet backbone: {backbone}")
