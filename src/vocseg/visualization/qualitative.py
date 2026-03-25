"""Qualitative analysis and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from vocseg.constants import DEFAULT_CLASS_COLORS, IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD
from vocseg.utils.io import ensure_dir


def denormalize_image_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    image = image_tensor * std + mean
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).cpu().numpy()


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(DEFAULT_CLASS_COLORS):
        color_mask[mask == class_id] = color
    color_mask[mask == IGNORE_INDEX] = (255, 255, 255)
    return color_mask


def save_sample_assets(
    image_tensor: torch.Tensor,
    ground_truth: torch.Tensor | np.ndarray,
    prediction: torch.Tensor | np.ndarray,
    output_dir: str | Path,
    image_id: str,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    inputs_dir = ensure_dir(output_dir / "inputs")
    gt_dir = ensure_dir(output_dir / "ground_truth")
    pred_dir = ensure_dir(output_dir / "predictions")
    triptych_dir = ensure_dir(output_dir / "triptychs")

    image = (denormalize_image_tensor(image_tensor) * 255).astype(np.uint8)
    gt = ground_truth.cpu().numpy() if isinstance(ground_truth, torch.Tensor) else np.asarray(ground_truth)
    pred = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else np.asarray(prediction)

    Image.fromarray(image).save(inputs_dir / f"{image_id}.png")
    Image.fromarray(mask_to_color(gt)).save(gt_dir / f"{image_id}.png")
    Image.fromarray(mask_to_color(pred)).save(pred_dir / f"{image_id}.png")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(mask_to_color(gt))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(mask_to_color(pred))
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    triptych_path = triptych_dir / f"{image_id}.png"
    figure.savefig(triptych_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return {
        "input_path": str(inputs_dir / f"{image_id}.png"),
        "ground_truth_path": str(gt_dir / f"{image_id}.png"),
        "prediction_path": str(pred_dir / f"{image_id}.png"),
        "triptych_path": str(triptych_path),
    }


def _tile_images(image_paths: Iterable[Path], output_path: str | Path, title: str, columns: int) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = int(np.ceil(len(images) / columns))
    canvas = Image.new("RGB", (columns * width, rows * height), color=(255, 255, 255))
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        canvas.paste(image.resize((width, height)), (col * width, row * height))

    figure = plt.figure(figsize=(4 * columns, 4 * rows))
    plt.imshow(canvas)
    plt.title(title)
    plt.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_best_worst_triptychs(per_image_df: pd.DataFrame, eval_dir: str | Path, output_path: str | Path, top_k: int = 3) -> None:
    eval_dir = Path(eval_dir)
    sorted_df = per_image_df.sort_values("image_miou", ascending=False)
    best = sorted_df.head(top_k)
    worst = sorted_df.tail(top_k)
    paths = [eval_dir / "qualitative" / "triptychs" / f"{row.image_id}.png" for row in pd.concat([best, worst]).itertuples()]
    _tile_images(paths, output_path, title="Top-3 Best and Worst Predictions", columns=top_k)


def plot_person_panel(per_image_df: pd.DataFrame, eval_dir: str | Path, output_path: str | Path, count: int = 6) -> None:
    eval_dir = Path(eval_dir)
    subset = per_image_df[per_image_df["person_present"]].sort_values("image_miou", ascending=True).head(count)
    paths = [eval_dir / "qualitative" / "triptychs" / f"{row.image_id}.png" for row in subset.itertuples()]
    _tile_images(paths, output_path, title="Person-Class Failure Cases", columns=min(3, max(len(paths), 1)))


def plot_per_class_iou(per_class_df: pd.DataFrame, output_path: str | Path) -> None:
    figure = plt.figure(figsize=(12, 5))
    plt.bar(per_class_df["class_name"], per_class_df["iou"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("IoU")
    plt.title("Per-Class IoU")
    plt.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_runtime_vs_accuracy(results_df: pd.DataFrame, output_path: str | Path) -> None:
    figure = plt.figure(figsize=(7, 5))
    plt.scatter(results_df["training_time_seconds"], results_df["mIoU"], s=120)
    for row in results_df.itertuples():
        plt.annotate(row.run_name, (row.training_time_seconds, row.mIoU))
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("mIoU")
    plt.title("Runtime vs Accuracy")
    plt.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
