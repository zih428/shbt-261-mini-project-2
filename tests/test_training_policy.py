from copy import deepcopy

import torch

from vocseg.config import dump_config
from vocseg.training.engine import build_scheduler, step_scheduler
from vocseg.training.runner import (
    _normalize_best_metrics_epoch,
    _normalize_history_epochs,
    run_artifacts_are_compatible,
)


def _sample_config() -> dict:
    return {
        "experiment_name": "demo_run",
        "runtime": {"amp": False},
        "data": {
            "metadata_dir": "/tmp/metadata",
            "batch_size": 8,
            "eval_batch_size": 4,
        },
        "model": {"family": "unet", "backbone": "resnet34", "pretrained": True},
        "optimizer": {"name": "adamw", "lr": 3e-4, "weight_decay": 1e-4, "betas": [0.9, 0.999]},
        "scheduler": {
            "name": "plateau",
            "max_epochs": 150,
            "factor": 0.5,
            "patience": 5,
            "threshold": 0.001,
            "min_lr": 1e-6,
        },
        "training": {
            "policy_version": 2,
            "monitor": "mIoU",
            "min_epochs": 20,
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.002,
        },
        "loss": {"name": "ce_dice"},
        "metrics": {"hd95_include_background": False},
    }


def test_plateau_scheduler_reduces_learning_rate():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = build_scheduler(
        optimizer,
        {
            "name": "plateau",
            "max_epochs": 150,
            "factor": 0.5,
            "patience": 0,
            "threshold": 0.0,
            "min_lr": 1e-6,
        },
    )

    step_scheduler(scheduler, metric_value=0.5)
    step_scheduler(scheduler, metric_value=0.4)

    assert optimizer.param_groups[0]["lr"] == 0.05


def test_resume_compatibility_ignores_metadata_dir(tmp_path):
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "runs" / "demo_run"
    run_dir.mkdir(parents=True)

    saved_config = _sample_config()
    saved_config["data"]["metadata_dir"] = "/tmp/old-metadata"
    dump_config(saved_config, run_dir / "resolved_config.yaml")

    current_config = _sample_config()
    current_config["data"]["metadata_dir"] = "/tmp/new-metadata"
    current_config["_config_path"] = "/tmp/current.yaml"

    assert run_artifacts_are_compatible(current_config, artifact_root)


def test_resume_compatibility_rejects_old_training_policy(tmp_path):
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "runs" / "demo_run"
    run_dir.mkdir(parents=True)

    saved_config = _sample_config()
    saved_config["scheduler"]["max_epochs"] = 60
    saved_config["training"]["policy_version"] = 1
    dump_config(saved_config, run_dir / "resolved_config.yaml")

    current_config = deepcopy(_sample_config())

    assert not run_artifacts_are_compatible(current_config, artifact_root)


def test_resume_normalizes_zero_based_history_epochs():
    history = [
        {"epoch": 0, "dev_mIoU": 0.10},
        {"epoch": 1, "dev_mIoU": 0.15},
        {"epoch": 2, "dev_mIoU": 0.20},
    ]

    normalized = _normalize_history_epochs(history)

    assert [row["epoch"] for row in normalized] == [1, 2, 3]


def test_resume_normalizes_best_metrics_epoch_from_history():
    history = [
        {"epoch": 1, "dev_mIoU": 0.10},
        {"epoch": 2, "dev_mIoU": 0.15},
        {"epoch": 3, "dev_mIoU": 0.20},
    ]
    best_metrics = {"epoch": 2, "mIoU": 0.20}

    normalized = _normalize_best_metrics_epoch(best_metrics, history, monitor="mIoU")

    assert normalized == {"epoch": 3, "mIoU": 0.20}
