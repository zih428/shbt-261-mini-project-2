#!/usr/bin/env python3
"""Prepare Pascal VOC metadata and the internal train/dev split."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import VOCSegmentation

from vocseg.data.metadata import (
    build_class_presence_matrix,
    build_metadata_for_dataset,
    finalize_metadata,
    iterative_multilabel_split,
    save_metadata_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root directory containing the VOCdevkit tree.")
    parser.add_argument("--output-dir", required=True, help="Directory to save metadata and split artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Internal dev fraction from official train.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dataset = VOCSegmentation(root=args.data_root, year="2007", image_set="train", download=False)
    val_dataset = VOCSegmentation(root=args.data_root, year="2007", image_set="val", download=False)

    train_frame = build_metadata_for_dataset(train_dataset, official_split="train")
    val_frame = build_metadata_for_dataset(val_dataset, official_split="val")

    label_matrix = build_class_presence_matrix(train_frame)
    train_indices, dev_indices = iterative_multilabel_split(label_matrix, val_fraction=args.val_fraction, seed=args.seed)
    train_frame, val_frame, thresholds = finalize_metadata(train_frame, val_frame)
    save_metadata_artifacts(
        train_frame=train_frame,
        val_frame=val_frame,
        train_indices=train_indices,
        dev_indices=dev_indices,
        thresholds=thresholds,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        val_fraction=args.val_fraction,
    )

    print(f"Saved metadata and split manifest to {args.output_dir}")
    print(f"Official train samples: {len(train_frame)}")
    print(f"Internal train/dev split: {len(train_indices)} / {len(dev_indices)}")
    print(f"Official validation samples: {len(val_frame)}")


if __name__ == "__main__":
    main()
