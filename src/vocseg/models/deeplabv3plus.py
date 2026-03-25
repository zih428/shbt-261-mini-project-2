"""DeepLabV3+ implementation with ResNet backbones."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocseg.models.common import ConvBNReLU, build_resnet_encoder


class ASPPBranch(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        kernel_size = 1 if dilation == 1 else 3
        padding = 0 if dilation == 1 else dilation
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, rates: tuple[int, int, int] = (6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ASPPBranch(in_channels, out_channels, dilation=1),
                ASPPBranch(in_channels, out_channels, dilation=rates[0]),
                ASPPBranch(in_channels, out_channels, dilation=rates[1]),
                ASPPBranch(in_channels, out_channels, dilation=rates[2]),
            ]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        outputs = [branch(x) for branch in self.branches]
        outputs.append(pooled)
        x = torch.cat(outputs, dim=1)
        return self.project(x)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if backbone not in {"resnet18", "resnet34", "resnet50"}:
            raise ValueError(f"Unsupported backbone for DeepLabV3+: {backbone}")

        dilation_setting = (False, False, True) if backbone == "resnet50" else None
        self.backbone = build_resnet_encoder(
            backbone=backbone,
            pretrained=pretrained,
            replace_stride_with_dilation=dilation_setting,
        )

        low_level_channels = 256 if backbone == "resnet50" else 64
        high_level_channels = 2048 if backbone == "resnet50" else 512

        self.aspp = ASPP(high_level_channels, out_channels=256)
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvBNReLU(256 + 48, 256),
            ConvBNReLU(256, 256),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def _extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        low = self.backbone.layer1(x)
        x = self.backbone.layer2(low)
        x = self.backbone.layer3(x)
        high = self.backbone.layer4(x)
        return low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        low, high = self._extract_features(x)
        x = self.aspp(high)
        low = self.low_level_projection(low)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
