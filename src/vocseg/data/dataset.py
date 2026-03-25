"""Dataset wrappers for Pascal VOC experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation

from vocseg.data.transforms import build_eval_transforms, build_train_transforms
from vocseg.utils.io import load_json


class VOCSegmentationExperimentDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        official_split: str,
        metadata_csv: str | Path,
        indices: list[int] | None = None,
        year: str = "2007",
        transform: Any | None = None,
    ):
        self.data_root = str(data_root)
        self.official_split = official_split
        self.transform = transform
        self.dataset = VOCSegmentation(
            root=self.data_root,
            year=year,
            image_set=official_split,
            download=False,
        )
        self.indices = indices if indices is not None else list(range(len(self.dataset)))
        metadata = pd.read_csv(metadata_csv)
        metadata["classes_present"] = metadata["classes_present"].fillna("").apply(
            lambda value: [int(item) for item in str(value).split(",") if item != ""]
        )
        metadata["subset_tags"] = metadata["subset_tags"].fillna("").apply(
            lambda value: [item for item in str(value).split(",") if item != ""]
        )
        self.metadata_by_index = metadata.set_index("dataset_index").to_dict(orient="index")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw_index = self.indices[index]
        image, mask = self.dataset[raw_index]
        row = dict(self.metadata_by_index[raw_index])
        row["dataset_index"] = raw_index
        row["official_split"] = self.official_split
        row["original_size"] = [image.height, image.width]

        if self.transform is not None:
            image, mask, transform_meta = self.transform(image, mask)
            row.update(transform_meta)
        return {
            "image": image,
            "mask": mask.long() if isinstance(mask, torch.Tensor) else mask,
            "meta": row,
        }


def segmentation_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    metas = [item["meta"] for item in batch]
    return {"images": images, "masks": masks, "metas": metas}


def build_dataset_bundle(data_cfg: dict[str, Any], data_root: str | Path, metadata_dir: str | Path) -> dict[str, VOCSegmentationExperimentDataset]:
    metadata_dir = Path(metadata_dir)
    manifest = load_json(metadata_dir / "split_manifest.json")
    train_transform = build_train_transforms(data_cfg)
    eval_transform = build_eval_transforms(data_cfg)

    bundle = {
        "internal_train": VOCSegmentationExperimentDataset(
            data_root=data_root,
            official_split="train",
            metadata_csv=metadata_dir / "train_metadata.csv",
            indices=manifest["internal_train"],
            transform=train_transform,
        ),
        "internal_dev": VOCSegmentationExperimentDataset(
            data_root=data_root,
            official_split="train",
            metadata_csv=metadata_dir / "train_metadata.csv",
            indices=manifest["internal_dev"],
            transform=eval_transform,
        ),
        "official_val": VOCSegmentationExperimentDataset(
            data_root=data_root,
            official_split="val",
            metadata_csv=metadata_dir / "val_metadata.csv",
            indices=manifest["official_val"],
            transform=eval_transform,
        ),
    }
    return bundle
