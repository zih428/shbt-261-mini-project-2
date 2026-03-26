"""Experiment runner."""

from __future__ import annotations

import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from vocseg.config import dump_config, load_config
from vocseg.constants import NUM_CLASSES
from vocseg.data import build_dataset_bundle, segmentation_collate_fn
from vocseg.models import build_model
from vocseg.training.engine import build_optimizer, build_scheduler, evaluate_model, step_scheduler, train_one_epoch
from vocseg.training.losses import build_loss
from vocseg.training.progress import compute_percent, load_history_if_exists, read_json_if_exists, utc_now_iso, write_progress_file
from vocseg.training.suite import load_suite_state, save_suite_state
from vocseg.utils.distributed import DistributedState, barrier, is_main_process
from vocseg.utils.io import ensure_dir, save_dataframe, save_json
from vocseg.utils.model_stats import count_parameters


def _make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    distributed: DistributedState,
) -> DataLoader:
    sampler = None
    if distributed.enabled:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=segmentation_collate_fn,
    )


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _resolve_device(distributed_state: DistributedState) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", distributed_state.local_rank)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _suite_prefix(progress_context: dict[str, Any] | None) -> str:
    if not progress_context:
        return ""
    run_index = progress_context.get("run_index")
    total_runs = progress_context.get("total_runs")
    if run_index is None or total_runs is None:
        return ""
    return f"[model {run_index}/{total_runs}] "


def _extract_training_policy(config: dict[str, Any]) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    scheduler_cfg = config.get("scheduler", {})
    return {
        "policy_version": int(training_cfg.get("policy_version", 1)),
        "monitor": str(training_cfg.get("monitor", "mIoU")),
        "min_epochs": int(training_cfg.get("min_epochs", 0)),
        "early_stopping_patience": int(training_cfg.get("early_stopping_patience", 10)),
        "early_stopping_min_delta": float(training_cfg.get("early_stopping_min_delta", 0.0)),
        "scheduler": deepcopy(scheduler_cfg),
    }


def _normalize_resume_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    normalized.pop("_config_path", None)
    data_cfg = normalized.get("data")
    if isinstance(data_cfg, dict):
        data_cfg = deepcopy(data_cfg)
        data_cfg.pop("metadata_dir", None)
        normalized["data"] = data_cfg
    return normalized


def _load_saved_run_config(run_dir: Path) -> dict[str, Any] | None:
    resolved_config_path = run_dir / "resolved_config.yaml"
    if resolved_config_path.exists():
        return _normalize_resume_config(load_config(resolved_config_path))

    checkpoint_last_path = run_dir / "checkpoint_last.pth"
    if not checkpoint_last_path.exists():
        return None

    checkpoint = torch.load(checkpoint_last_path, map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        return None
    return _normalize_resume_config(saved_config)


def run_artifacts_are_compatible(config: dict[str, Any], artifact_root: str | Path) -> bool:
    run_dir = Path(artifact_root) / "runs" / config["experiment_name"]
    saved_config = _load_saved_run_config(run_dir)
    if saved_config is None:
        return True
    return saved_config == _normalize_resume_config(config)


def _archive_incompatible_run_dir(run_dir: Path, artifact_root: Path) -> Path:
    archive_root = ensure_dir(artifact_root / "runs_archive")
    timestamp = utc_now_iso().replace(":", "-")
    archive_path = archive_root / f"{run_dir.name}__{timestamp}"
    suffix = 1
    while archive_path.exists():
        archive_path = archive_root / f"{run_dir.name}__{timestamp}_{suffix}"
        suffix += 1
    shutil.move(str(run_dir), str(archive_path))
    return archive_path


def _prepare_run_dir(
    *,
    config: dict[str, Any],
    artifact_root: Path,
    distributed_state: DistributedState,
    resume: bool,
    progress_context: dict[str, Any] | None,
    print_progress: bool,
) -> Path:
    experiment_name = config["experiment_name"]
    run_dir = artifact_root / "runs" / experiment_name
    archive_path: Path | None = None
    if resume and run_dir.exists() and not run_artifacts_are_compatible(config, artifact_root):
        if is_main_process(distributed_state):
            archive_path = _archive_incompatible_run_dir(run_dir, artifact_root)
        barrier(distributed_state)
        if is_main_process(distributed_state) and print_progress:
            prefix = _suite_prefix(progress_context)
            print(
                f"{prefix}{experiment_name}: archived stale run artifacts to {archive_path} "
                "because the training config changed."
            )
    return ensure_dir(run_dir)


def _resolve_monitor_value(metrics: dict[str, Any], monitor: str) -> float:
    key = monitor.removeprefix("dev_")
    if key not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise KeyError(f"Monitor '{monitor}' not found in validation summary. Available metrics: {available}")
    return float(metrics[key])


def _normalize_history_epochs(train_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not train_history:
        return train_history

    epoch_values: list[int] = []
    for row in train_history:
        epoch_value = row.get("epoch")
        if epoch_value is None or pd.isna(epoch_value):
            return train_history
        epoch_values.append(int(epoch_value))

    expected_zero_based = list(range(len(train_history)))
    if epoch_values != expected_zero_based:
        return train_history

    normalized_history: list[dict[str, Any]] = []
    for index, row in enumerate(train_history, start=1):
        normalized_row = dict(row)
        normalized_row["epoch"] = index
        normalized_history.append(normalized_row)
    return normalized_history


def _normalize_best_metrics_epoch(
    best_metrics: dict[str, Any] | None,
    train_history: list[dict[str, Any]],
    monitor: str,
) -> dict[str, Any] | None:
    if not best_metrics or not train_history:
        return best_metrics

    metric_key = monitor.removeprefix("dev_")
    best_value = best_metrics.get(metric_key)
    if best_value is None:
        return best_metrics

    history_key = f"dev_{metric_key}"
    matching_epochs: list[int] = []
    for row in train_history:
        history_value = row.get(history_key)
        if history_value is None or pd.isna(history_value):
            continue
        if abs(float(history_value) - float(best_value)) <= 1e-12:
            matching_epochs.append(int(row["epoch"]))

    if not matching_epochs:
        return best_metrics

    normalized_metrics = deepcopy(best_metrics)
    saved_epoch = best_metrics.get("epoch")
    if saved_epoch is not None and not pd.isna(saved_epoch):
        saved_epoch_int = int(saved_epoch)
        zero_based_candidate = saved_epoch_int + 1
        if zero_based_candidate in matching_epochs:
            normalized_metrics["epoch"] = zero_based_candidate
            return normalized_metrics
        if saved_epoch_int in matching_epochs:
            normalized_metrics["epoch"] = saved_epoch_int
            return normalized_metrics

    normalized_metrics["epoch"] = matching_epochs[0]
    return normalized_metrics


def _update_suite_run_state(
    progress_context: dict[str, Any] | None,
    run_name: str,
    status: str,
    current_epoch: int,
    max_epochs: int,
    percent_complete: float,
    best_dev_miou: float | None,
    run_dir: Path,
) -> None:
    if not progress_context or "suite_state_path" not in progress_context:
        return
    suite_state_path = Path(progress_context["suite_state_path"])
    state = load_suite_state(suite_state_path)
    if state is None:
        return
    run_index = int(progress_context["run_index"])
    for run in state["runs"]:
        if int(run["index"]) != run_index:
            continue
        run["status"] = status
        run["current_epoch"] = current_epoch
        run["max_epochs"] = max_epochs
        run["percent_complete"] = percent_complete
        run["run_dir"] = str(run_dir)
        run["best_dev_mIoU"] = best_dev_miou
        run["last_update"] = utc_now_iso()
        break
    state["status"] = status if status in {"completed", "failed", "interrupted"} and progress_context.get("total_runs", 1) == 1 else "running"
    save_suite_state(suite_state_path, state)


def _write_run_progress(
    run_dir: Path,
    *,
    config: dict[str, Any],
    status: str,
    current_epoch: int,
    max_epochs: int,
    best_metrics: dict[str, Any] | None,
    resumed_from_epoch: int,
    elapsed_training_time_seconds: float,
    epochs_without_improvement: int,
    progress_context: dict[str, Any] | None,
    latest_epoch_metrics: dict[str, Any] | None,
) -> None:
    progress_path = run_dir / "progress.json"
    existing = read_json_if_exists(progress_path) or {}
    percent_complete = compute_percent(current_epoch, max_epochs)
    if status == "completed":
        percent_complete = 100.0
    payload = {
        "run_name": config["experiment_name"],
        "config_path": config.get("_config_path"),
        "status": status,
        "current_epoch": current_epoch,
        "max_epochs": max_epochs,
        "percent_complete": percent_complete,
        "resumed_from_epoch": resumed_from_epoch,
        "elapsed_training_time_seconds": elapsed_training_time_seconds,
        "epochs_without_improvement": epochs_without_improvement,
        "training_policy": _extract_training_policy(config),
        "best_dev_metrics": best_metrics or {},
        "latest_epoch_metrics": latest_epoch_metrics or {},
        "updated_at": utc_now_iso(),
        "started_at": existing.get("started_at", utc_now_iso()),
        "suite": {
            "run_index": progress_context.get("run_index") if progress_context else None,
            "total_runs": progress_context.get("total_runs") if progress_context else None,
            "suite_name": progress_context.get("suite_name") if progress_context else None,
        },
    }
    write_progress_file(progress_path, payload)
    best_miou = None if not best_metrics else best_metrics.get("mIoU")
    _update_suite_run_state(
        progress_context=progress_context,
        run_name=config["experiment_name"],
        status=status,
        current_epoch=current_epoch,
        max_epochs=max_epochs,
        percent_complete=payload["percent_complete"],
        best_dev_miou=best_miou,
        run_dir=run_dir,
    )


def fit(
    config: dict[str, Any],
    data_root: str | Path,
    artifact_root: str | Path,
    distributed_state: DistributedState,
    evaluation_split: str = "internal_dev",
    resume: bool = True,
    progress_context: dict[str, Any] | None = None,
    print_progress: bool = True,
) -> Path:
    experiment_name = config["experiment_name"]
    artifact_root = Path(artifact_root)
    run_dir = _prepare_run_dir(
        config=config,
        artifact_root=artifact_root,
        distributed_state=distributed_state,
        resume=resume,
        progress_context=progress_context,
        print_progress=print_progress,
    )
    metadata_dir = config["data"]["metadata_dir"]
    dataset_bundle = build_dataset_bundle(config["data"], data_root=data_root, metadata_dir=metadata_dir)

    train_dataset = dataset_bundle["internal_train"]
    dev_dataset = dataset_bundle[evaluation_split]

    batch_size = int(config["data"]["batch_size"])
    eval_batch_size = int(config["data"].get("eval_batch_size", batch_size))
    num_workers = int(config["data"].get("num_workers", 4))

    train_loader = _make_loader(train_dataset, batch_size, num_workers, shuffle=True, distributed=distributed_state)
    dev_loader = _make_loader(dev_dataset, eval_batch_size, num_workers, shuffle=False, distributed=distributed_state)

    device = _resolve_device(distributed_state)
    model = build_model(config["model"], num_classes=NUM_CLASSES).to(device)
    if distributed_state.enabled:
        model = DistributedDataParallel(model, device_ids=[distributed_state.local_rank] if device.type == "cuda" else None)

    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(optimizer, config["scheduler"])
    loss_fn = build_loss(config["loss"])
    use_amp = bool(config["runtime"].get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    monitor = str(config["training"].get("monitor", "mIoU"))
    min_epochs = int(config["training"].get("min_epochs", 0))
    max_epochs = int(config["scheduler"]["max_epochs"])
    patience = int(config["training"].get("early_stopping_patience", 10))
    min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    train_log_path = run_dir / "train_log.csv"
    checkpoint_last_path = run_dir / "checkpoint_last.pth"
    checkpoint_best_path = run_dir / "checkpoint_best.pth"

    if is_main_process(distributed_state):
        dump_config(config, run_dir / "resolved_config.yaml")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    completed_metrics_path = run_dir / "metrics.json"
    existing_progress = read_json_if_exists(run_dir / "progress.json") or {}
    if resume and completed_metrics_path.exists() and existing_progress.get("status") == "completed":
        if is_main_process(distributed_state) and print_progress:
            prefix = _suite_prefix(progress_context)
            print(f"{prefix}{experiment_name}: already completed, skipping.")
        return run_dir

    start_time = time.perf_counter()
    train_history = load_history_if_exists(train_log_path) if resume else []
    train_history = _normalize_history_epochs(train_history)
    best_score = float("-inf")
    epochs_without_improvement = 0
    best_metrics: dict[str, Any] | None = None
    start_epoch = 0
    elapsed_before_resume = 0.0
    resumed_from_epoch = 0

    if resume and checkpoint_last_path.exists():
        checkpoint = torch.load(checkpoint_last_path, map_location=device, weights_only=False)
        _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler_state_dict = checkpoint.get("scaler_state_dict")
        if scaler_state_dict is not None and use_amp and device.type == "cuda":
            scaler.load_state_dict(scaler_state_dict)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        resumed_from_epoch = start_epoch
        best_score = float(checkpoint.get("best_score", float("-inf")))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        best_metrics = checkpoint.get("best_metrics")
        best_metrics = _normalize_best_metrics_epoch(best_metrics, train_history, monitor)
        elapsed_before_resume = float(checkpoint.get("elapsed_training_time_seconds", 0.0))
        if is_main_process(distributed_state) and print_progress:
            prefix = _suite_prefix(progress_context)
            print(f"{prefix}{experiment_name}: resuming at epoch {start_epoch + 1}/{max_epochs}.")

    if is_main_process(distributed_state):
        initial_status = "resuming" if start_epoch > 0 else "running"
        _write_run_progress(
            run_dir,
            config=config,
            status=initial_status,
            current_epoch=start_epoch,
            max_epochs=max_epochs,
            best_metrics=best_metrics,
            resumed_from_epoch=resumed_from_epoch,
            elapsed_training_time_seconds=elapsed_before_resume,
            epochs_without_improvement=epochs_without_improvement,
            progress_context=progress_context,
            latest_epoch_metrics=train_history[-1] if train_history else None,
        )

    for epoch in range(start_epoch, max_epochs):
        train_sampler = getattr(train_loader, "sampler", None)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            amp=use_amp,
            accumulation_steps=accumulation_steps,
        )

        dev_results = evaluate_model(
            model=model,
            loader=dev_loader,
            device=device,
            save_dir=None,
            save_qualitative_assets=False,
            include_background_hd95=bool(config["metrics"].get("hd95_include_background", False)),
        )
        dev_summary = dev_results["summary"]
        current_lr = float(optimizer.param_groups[0]["lr"])
        current_score = _resolve_monitor_value(dev_summary, monitor)

        completed_epochs = epoch + 1
        row = {
            "epoch": completed_epochs,
            "train_loss": train_stats["loss"],
            "train_images_per_second": train_stats["images_per_second"],
            "train_epoch_time_seconds": train_stats["epoch_time_seconds"],
            "dev_mIoU": dev_summary["mIoU"],
            "dev_mean_dice": dev_summary["mean_dice"],
            "dev_pixel_accuracy": dev_summary["pixel_accuracy"],
            "dev_hd95": dev_summary["hd95"],
            "lr": current_lr,
        }
        train_history.append(row)
        elapsed_training_time_seconds = elapsed_before_resume + (time.perf_counter() - start_time)

        if current_score > (best_score + min_delta):
            best_score = current_score
            best_metrics = {
                "epoch": completed_epochs,
                **dev_summary,
            }
            epochs_without_improvement = 0
            if is_main_process(distributed_state):
                torch.save(
                    {
                        "model_state_dict": _unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if use_amp and device.type == "cuda" else None,
                        "config": config,
                        "epoch": epoch,
                        "best_score": best_score,
                        "best_metrics": best_metrics,
                        "epochs_without_improvement": epochs_without_improvement,
                        "elapsed_training_time_seconds": elapsed_training_time_seconds,
                    },
                    checkpoint_best_path,
                )
        else:
            epochs_without_improvement += 1

        step_scheduler(scheduler, metric_value=current_score)

        if is_main_process(distributed_state):
            torch.save(
                    {
                        "model_state_dict": _unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if use_amp and device.type == "cuda" else None,
                        "config": config,
                        "epoch": epoch,
                        "best_score": best_score,
                        "best_metrics": best_metrics,
                        "epochs_without_improvement": epochs_without_improvement,
                        "elapsed_training_time_seconds": elapsed_training_time_seconds,
                    },
                    checkpoint_last_path,
                )
            history_df = pd.DataFrame(train_history)
            save_dataframe(history_df, train_log_path)
            avg_epoch_time = float(history_df["train_epoch_time_seconds"].mean()) if not history_df.empty else 0.0
            epochs_remaining = max(max_epochs - completed_epochs, 0)
            eta_seconds = avg_epoch_time * epochs_remaining
            row["eta_seconds"] = eta_seconds
            _write_run_progress(
                run_dir,
                config=config,
                status="running",
                current_epoch=completed_epochs,
                max_epochs=max_epochs,
                best_metrics=best_metrics,
                resumed_from_epoch=resumed_from_epoch,
                elapsed_training_time_seconds=elapsed_training_time_seconds,
                epochs_without_improvement=epochs_without_improvement,
                progress_context=progress_context,
                latest_epoch_metrics=row,
            )
            if print_progress:
                prefix = _suite_prefix(progress_context)
                best_value = None if not best_metrics else _resolve_monitor_value(best_metrics, monitor)
                best_text = "nan" if best_value is None else f"{best_value:.4f}"
                print(
                    f"{prefix}{experiment_name}: epoch {completed_epochs}/{max_epochs} "
                    f"train_loss={row['train_loss']:.4f} {monitor}={current_score:.4f} "
                    f"best_{monitor}={best_text} eta_s={eta_seconds:.0f}"
                )

        if (epoch + 1) >= min_epochs and epochs_without_improvement >= patience:
            break

    total_time = elapsed_before_resume + (time.perf_counter() - start_time)
    peak_memory = float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)) if device.type == "cuda" else 0.0

    if is_main_process(distributed_state):
        history_df = pd.DataFrame(train_history)
        save_dataframe(history_df, train_log_path)
        summary = {
            "run_name": experiment_name,
            "training_time_seconds": total_time,
            "peak_gpu_memory_gb": peak_memory,
            "total_parameters": count_parameters(_unwrap_model(model)),
            "trainable_parameters": count_parameters(_unwrap_model(model), trainable_only=True),
            "training_policy": _extract_training_policy(config),
            "best_dev_metrics": best_metrics or {},
        }
        save_json(summary, run_dir / "metrics.json")
        _write_run_progress(
            run_dir,
            config=config,
            status="completed",
            current_epoch=len(train_history),
            max_epochs=max_epochs,
            best_metrics=best_metrics,
            resumed_from_epoch=resumed_from_epoch,
            elapsed_training_time_seconds=total_time,
            epochs_without_improvement=epochs_without_improvement,
            progress_context=progress_context,
            latest_epoch_metrics=train_history[-1] if train_history else None,
        )
        if print_progress:
            prefix = _suite_prefix(progress_context)
            best_value = None if not best_metrics else _resolve_monitor_value(best_metrics, monitor)
            best_text = "nan" if best_value is None else f"{best_value:.4f}"
            print(f"{prefix}{experiment_name}: completed. best_{monitor}={best_text}")

    return run_dir
