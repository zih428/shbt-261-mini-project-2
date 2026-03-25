"""SegFormer wrapper built on timm MiT backbones."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None


class SegFormerHead(nn.Module):
    def __init__(self, in_channels: list[int], embedding_dim: int, num_classes: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(channels, embedding_dim, kernel_size=1) for channels in in_channels]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * len(in_channels), embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        projected = []
        target_size = features[0].shape[-2:]
        for projection, feature in zip(self.projections, features):
            x = projection(feature)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)
        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class SegFormer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "mit_b2",
        pretrained: bool = True,
        embedding_dim: int = 256,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("SegFormer requires timm. Install `timm` to use this model.")
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.decode_head = SegFormerHead(
            in_channels=list(self.backbone.feature_info.channels()),
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.decode_head(features, output_size=x.shape[-2:])
