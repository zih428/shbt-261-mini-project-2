"""SegFormer wrapper built on Hugging Face Transformers checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from huggingface_hub import snapshot_download
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except ImportError:  # pragma: no cover - optional dependency
    snapshot_download = None
    SegformerConfig = None
    SegformerForSemanticSegmentation = None


SEGFORMER_MODEL_IDS = {
    "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "mit_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "mit_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "mit_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}

SEGFORMER_ARCH_CONFIGS = {
    "segformer_b0": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [32, 64, 160, 256],
        "decoder_hidden_size": 256,
    },
    "segformer_b1": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 256,
    },
    "segformer_b2": {
        "depths": [3, 4, 6, 3],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 256,
    },
}


def _canonical_variant(backbone_name: str) -> str:
    if backbone_name.startswith("nvidia/"):
        if "segformer-b0" in backbone_name or "mit-b0" in backbone_name:
            return "segformer_b0"
        if "segformer-b1" in backbone_name or "mit-b1" in backbone_name:
            return "segformer_b1"
        if "segformer-b2" in backbone_name or "mit-b2" in backbone_name:
            return "segformer_b2"
    if backbone_name not in SEGFORMER_MODEL_IDS:
        raise ValueError(f"Unsupported SegFormer backbone: {backbone_name}")
    return backbone_name.replace("mit_", "segformer_")


def _resolve_model_id(backbone_name: str) -> str:
    if backbone_name.startswith("nvidia/"):
        return backbone_name
    return SEGFORMER_MODEL_IDS[backbone_name]


class SegFormer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "segformer_b2",
        pretrained: bool = True,
        embedding_dim: int = 256,
    ):
        super().__init__()
        if snapshot_download is None or SegformerConfig is None or SegformerForSemanticSegmentation is None:
            raise ImportError(
                "SegFormer requires `transformers`. Install the models extra with `uv sync --group dev --extra models`."
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
                        "model.safetensors",
                        "model.safetensors.index.json",
                        "pytorch_model.bin",
                    ],
                )
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    local_snapshot,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    local_files_only=True,
                    use_safetensors=True,
                )
            except OSError:
                try:
                    local_snapshot = snapshot_download(
                        repo_id=model_id,
                        local_files_only=True,
                        allow_patterns=[
                            "config.json",
                            "preprocessor_config.json",
                            "pytorch_model.bin",
                        ],
                    )
                    self.model = SegformerForSemanticSegmentation.from_pretrained(
                        local_snapshot,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True,
                        local_files_only=True,
                        use_safetensors=False,
                    )
                except OSError as inner_exc:
                    raise RuntimeError(
                        "Failed to load cached SegFormer weights. "
                        "Run `uv run python scripts/fetch_pretrained_assets.py` first."
                    ) from inner_exc
        else:
            arch = SEGFORMER_ARCH_CONFIGS[canonical_variant]
            config = SegformerConfig(
                num_labels=num_classes,
                depths=arch["depths"],
                hidden_sizes=arch["hidden_sizes"],
                decoder_hidden_size=max(int(embedding_dim), int(arch["decoder_hidden_size"])),
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                mlp_ratios=[4, 4, 4, 4],
                hidden_act="gelu",
                reshape_last_stage=True,
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
