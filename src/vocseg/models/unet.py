"""U-Net with ResNet encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocseg.models.common import ConvBNReLU, build_resnet_encoder, extract_resnet_feature_pyramid, infer_resnet_channels


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetResNet(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.encoder = build_resnet_encoder(backbone=backbone, pretrained=pretrained)
        channels = infer_resnet_channels(backbone)

        self.center = nn.Sequential(
            ConvBNReLU(channels[4], 512),
            ConvBNReLU(512, 512),
        )
        self.decoder4 = DecoderBlock(512, channels[3], 256)
        self.decoder3 = DecoderBlock(256, channels[2], 128)
        self.decoder2 = DecoderBlock(128, channels[1], 96)
        self.decoder1 = DecoderBlock(96, channels[0], 64)
        self.classifier = nn.Sequential(
            ConvBNReLU(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0, x1, x2, x3, x4 = extract_resnet_feature_pyramid(self.encoder, x)
        x = self.center(x4)
        x = self.decoder4(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder2(x, x1)
        x = self.decoder1(x, x0)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
