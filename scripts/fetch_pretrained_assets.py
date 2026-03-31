#!/usr/bin/env python3
"""Download pretrained assets needed by the experiment configs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import torch
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


TORCHVISION_WEIGHTS = {
    "resnet18": ResNet18_Weights.IMAGENET1K_V1.url,
    "resnet34": ResNet34_Weights.IMAGENET1K_V1.url,
    "resnet50": ResNet50_Weights.IMAGENET1K_V2.url,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resnets",
        nargs="*",
        default=["resnet18", "resnet34", "resnet50"],
        choices=sorted(TORCHVISION_WEIGHTS),
        help="Torchvision ResNet checkpoints to cache locally.",
    )
    parser.add_argument(
        "--segformer-model-id",
        default="nvidia/segformer-b2-finetuned-ade-512-512",
        help="Hugging Face model id for the SegFormer checkpoint.",
    )
    parser.add_argument(
        "--sam2-model-id",
        default="facebook/sam2-hiera-small",
        help="Hugging Face model id for the SAM2 checkpoint.",
    )
    parser.add_argument(
        "--mask2former-model-id",
        default="facebook/mask2former-swin-tiny-ade-semantic",
        help="Hugging Face model id for the Mask2Former checkpoint.",
    )
    parser.add_argument("--skip-segformer", action="store_true", help="Do not prefetch SegFormer weights.")
    parser.add_argument("--skip-sam2", action="store_true", help="Do not prefetch SAM2 weights.")
    parser.add_argument("--skip-mask2former", action="store_true", help="Do not prefetch Mask2Former weights.")
    return parser.parse_args()


def _download_torchvision_weight(url: str) -> Path:
    checkpoints_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoints_dir / Path(urlparse(url).path).name
    if destination.exists():
        print(f"torchvision weight already cached: {destination}")
        return destination
    subprocess.run(["curl", "-L", "-k", url, "-o", str(destination)], check=True)
    print(f"cached torchvision weight: {destination}")
    return destination


def _prefetch_segformer(model_id: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
        ],
    )
    print(f"cached SegFormer weights: {snapshot_path}")


def _prefetch_sam2(model_id: str) -> None:
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
    from huggingface_hub import hf_hub_download

    _, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    print(f"cached SAM2 weights: {path}")


def _prefetch_mask2former(model_id: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
        ],
    )
    print(f"cached Mask2Former weights: {snapshot_path}")


def main() -> None:
    args = parse_args()
    for backbone in args.resnets:
        _download_torchvision_weight(TORCHVISION_WEIGHTS[backbone])
    if not args.skip_segformer:
        _prefetch_segformer(args.segformer_model_id)
    if not args.skip_sam2:
        _prefetch_sam2(args.sam2_model_id)
    if not args.skip_mask2former:
        _prefetch_mask2former(args.mask2former_model_id)


if __name__ == "__main__":
    main()
