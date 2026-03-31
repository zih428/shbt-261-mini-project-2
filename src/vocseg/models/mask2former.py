"""Mask2Former wrapper built on Hugging Face Transformers checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from huggingface_hub import snapshot_download
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig
except ImportError:  # pragma: no cover - optional dependency
    snapshot_download = None
    Mask2FormerConfig = None
    Mask2FormerForUniversalSegmentation = None
    SwinConfig = None


MASK2FORMER_MODEL_IDS = {
    "mask2former_swin_tiny": "facebook/mask2former-swin-tiny-ade-semantic",
    "mask2former_swin_small": "facebook/mask2former-swin-small-ade-semantic",
    "mask2former_swin_base": "facebook/mask2former-swin-base-ade-semantic",
}

MASK2FORMER_BACKBONE_CONFIGS = {
    "mask2former_swin_tiny": {
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "drop_path_rate": 0.2,
    },
    "mask2former_swin_small": {
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "drop_path_rate": 0.3,
    },
    "mask2former_swin_base": {
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "drop_path_rate": 0.3,
    },
}


def _canonical_variant(backbone_name: str) -> str:
    if backbone_name.startswith("facebook/"):
        if "swin-tiny" in backbone_name:
            return "mask2former_swin_tiny"
        if "swin-small" in backbone_name:
            return "mask2former_swin_small"
        if "swin-base" in backbone_name:
            return "mask2former_swin_base"
    if backbone_name not in MASK2FORMER_MODEL_IDS:
        raise ValueError(f"Unsupported Mask2Former backbone: {backbone_name}")
    return backbone_name


def _resolve_model_id(backbone_name: str) -> str:
    if backbone_name.startswith("facebook/"):
        return backbone_name
    return MASK2FORMER_MODEL_IDS[backbone_name]


class Mask2Former(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "mask2former_swin_tiny",
        pretrained: bool = True,
    ):
        super().__init__()
        if (
            snapshot_download is None
            or Mask2FormerConfig is None
            or Mask2FormerForUniversalSegmentation is None
            or SwinConfig is None
        ):
            raise ImportError(
                "Mask2Former requires `transformers`. Install the models extra with `uv sync --group dev --extra models`."
            )

        canonical_variant = _canonical_variant(backbone_name)
        model_id = _resolve_model_id(backbone_name)

        if pretrained:
            try:
                local_snapshot = snapshot_download(
                    repo_id=model_id,
                    local_files_only=True,
                    allow_patterns=[
                        "config.json",
                        "preprocessor_config.json",
                        "processor_config.json",
                        "model.safetensors",
                        "model.safetensors.index.json",
                        "pytorch_model.bin",
                    ],
                )
                self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    local_snapshot,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    local_files_only=True,
                    use_safetensors=True,
                    output_auxiliary_logits=False,
                )
            except OSError:
                try:
                    local_snapshot = snapshot_download(
                        repo_id=model_id,
                        local_files_only=True,
                        allow_patterns=[
                            "config.json",
                            "preprocessor_config.json",
                            "processor_config.json",
                            "pytorch_model.bin",
                        ],
                    )
                    self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                        local_snapshot,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True,
                        local_files_only=True,
                        use_safetensors=False,
                        output_auxiliary_logits=False,
                    )
                except OSError as inner_exc:
                    raise RuntimeError(
                        "Failed to load cached Mask2Former weights. "
                        "Run `uv run python scripts/fetch_pretrained_assets.py --mask2former-model-id "
                        "facebook/mask2former-swin-tiny-ade-semantic` first."
                    ) from inner_exc
        else:
            backbone_cfg = MASK2FORMER_BACKBONE_CONFIGS[canonical_variant]
            config = Mask2FormerConfig(
                backbone_config=SwinConfig(
                    image_size=512,
                    embed_dim=int(backbone_cfg["embed_dim"]),
                    depths=list(backbone_cfg["depths"]),
                    num_heads=list(backbone_cfg["num_heads"]),
                    drop_path_rate=float(backbone_cfg["drop_path_rate"]),
                    out_features=["stage1", "stage2", "stage3", "stage4"],
                ),
                num_labels=num_classes,
                hidden_dim=256,
                feature_size=256,
                mask_feature_size=256,
                use_auxiliary_loss=False,
                output_auxiliary_logits=False,
            )
            self.model = Mask2FormerForUniversalSegmentation(config)

        self.model.config.use_auxiliary_loss = False
        self.model.config.output_auxiliary_logits = False

    @staticmethod
    def _semantic_logits(outputs, target_size: tuple[int, int]) -> torch.Tensor:
        class_queries_logits = outputs.class_queries_logits[..., :-1].softmax(dim=-1)
        masks_queries_logits = F.interpolate(
            outputs.masks_queries_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        return torch.einsum("bqc,bqhw->bchw", class_queries_logits, masks_queries_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x, output_hidden_states=True, output_auxiliary_logits=False)
        return self._semantic_logits(outputs, target_size=tuple(x.shape[-2:]))
