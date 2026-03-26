"""Training and evaluation loops."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from vocseg.constants import NUM_CLASSES, VOC_CLASSES
from vocseg.evaluation.metrics import SegmentationMetricAccumulator
from vocseg.visualization.qualitative import save_sample_assets


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = True,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_images = 0
    start_time = time.perf_counter()

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, masks)

        if scaler is not None and amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_images += batch_size

    elapsed = time.perf_counter() - start_time
    return {
        "loss": total_loss / max(total_images, 1),
        "images_per_second": total_images / max(elapsed, 1e-8),
        "epoch_time_seconds": elapsed,
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str | Path | None = None,
    save_qualitative_assets: bool = False,
    include_background_hd95: bool = False,
) -> dict[str, Any]:
    model.eval()
    start_time = time.perf_counter()
    total_images = 0
    accumulator = SegmentationMetricAccumulator(
        num_classes=NUM_CLASSES,
        class_names=VOC_CLASSES,
        include_background_hd95=include_background_hd95,
    )
    save_dir = Path(save_dir) if save_dir is not None else None

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"]
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

        for batch_index, meta in enumerate(batch["metas"]):
            target_np = masks[batch_index].cpu().numpy().astype(np.int64)
            pred_np = predictions[batch_index].numpy().astype(np.int64)
            accumulator.update(pred_np, target_np, meta)

            if save_dir is not None and save_qualitative_assets:
                asset_paths = save_sample_assets(
                    image_tensor=batch["images"][batch_index].cpu(),
                    ground_truth=masks[batch_index].cpu(),
                    prediction=predictions[batch_index].cpu(),
                    output_dir=save_dir / "qualitative",
                    image_id=meta["image_id"],
                )
                accumulator.records[-1].update(asset_paths)

        total_images += images.shape[0]

    elapsed = time.perf_counter() - start_time
    summary, per_class, per_image, subset_metrics = accumulator.finalize()
    summary["inference_time_seconds"] = elapsed
    summary["images_per_second"] = total_images / max(elapsed, 1e-8)
    return {
        "summary": summary,
        "per_class": per_class,
        "per_image": per_image,
        "subset_metrics": subset_metrics,
    }


def build_optimizer(model: nn.Module, optimizer_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = optimizer_cfg.get("name", "adamw").lower()
    lr = float(optimizer_cfg.get("lr", 1e-4))
    weight_decay = float(optimizer_cfg.get("weight_decay", 1e-4))
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
    name = str(scheduler_cfg.get("name", "cosine")).lower()
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 5)),
            threshold=float(scheduler_cfg.get("threshold", 1e-3)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name}")

    max_epochs = int(scheduler_cfg["max_epochs"])
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = (epoch - warmup_epochs) / float(max(max_epochs - warmup_epochs, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau,
    metric_value: float | None = None,
) -> None:
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric_value is None:
            raise ValueError("Plateau scheduler requires a validation metric.")
        scheduler.step(metric_value)
        return
    scheduler.step()
