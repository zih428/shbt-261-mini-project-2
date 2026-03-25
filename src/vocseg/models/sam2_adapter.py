"""Optional SAM2 semantic segmentation adapter."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocseg.models.common import ConvBNReLU
from vocseg.models.lora import apply_lora_to_matching_modules


def _resolve_sam2_builder() -> Any:  # pragma: no cover - optional dependency
    candidates = [
        ("sam2.build_sam", "build_sam2"),
        ("sam2.build_sam2", "build_sam2"),
    ]
    for module_name, attr_name in candidates:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            return getattr(module, attr_name)
        except (ImportError, AttributeError):
            continue
    raise ImportError("Could not import SAM2. Install the SAM2 package and checkpoint dependencies.")


class SAM2SemanticDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvBNReLU(in_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = self.decoder(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class SAM2SemanticSegmentor(nn.Module):
    def __init__(self, num_classes: int, sam2_cfg: dict[str, Any]):
        super().__init__()
        builder = _resolve_sam2_builder()
        config_path = sam2_cfg["config_path"]
        checkpoint_path = sam2_cfg["checkpoint_path"]
        device = sam2_cfg.get("build_device", "cpu")
        self.sam_model = builder(config_path, checkpoint_path, device=device)
        self.image_encoder = self._resolve_image_encoder(self.sam_model)
        self.feature_channels = int(sam2_cfg.get("feature_channels", 256))

        freeze_backbone = sam2_cfg.get("freeze_backbone", True)
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = not freeze_backbone

        lora_cfg = sam2_cfg.get("lora", {})
        if lora_cfg.get("enabled", False):
            targets = lora_cfg.get("target_modules", [])
            rank = int(lora_cfg.get("rank", 8))
            alpha = float(lora_cfg.get("alpha", 16.0))
            patched = apply_lora_to_matching_modules(self.image_encoder, targets, rank=rank, alpha=alpha)
            if not patched:
                raise ValueError("LoRA was enabled for SAM2, but no target modules matched.")

        self.decoder = SAM2SemanticDecoder(
            in_channels=self.feature_channels,
            num_classes=num_classes,
            hidden_channels=int(sam2_cfg.get("decoder_channels", 256)),
        )

    @staticmethod
    def _resolve_image_encoder(sam_model: nn.Module) -> nn.Module:
        for candidate in ("image_encoder", "image_encoder_trunk", "vision_encoder", "backbone"):
            if hasattr(sam_model, candidate):
                return getattr(sam_model, candidate)
        raise AttributeError("Could not find a SAM2 image encoder on the built model.")

    def _extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(x)
        if isinstance(features, dict):
            for key in ("vision_features", "image_embeddings", "features", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
            else:
                features = next(reversed(features.values()))
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"SAM2 encoder output type is unsupported: {type(features)}")
        if features.ndim == 3:
            batch_size, token_count, channels = features.shape
            side = int(math.sqrt(token_count))
            if side * side != token_count:
                raise ValueError("SAM2 token sequence is not square; provide a custom adapter.")
            features = features.transpose(1, 2).reshape(batch_size, channels, side, side)
        if features.ndim != 4:
            raise ValueError("SAM2 encoder output must be a 4D feature map or a square token grid.")
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extract_feature_map(x)
        return self.decoder(features, output_size=x.shape[-2:])
