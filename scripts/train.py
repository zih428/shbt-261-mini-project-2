#!/usr/bin/env python3
"""Train a segmentation model on the internal Pascal VOC split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from vocseg.config import load_config
from vocseg.training.runner import fit
from vocseg.training.progress import read_json_if_exists, write_progress_file
from vocseg.utils.distributed import cleanup_distributed, init_distributed, is_main_process
from vocseg.utils.io import load_json
from vocseg.utils.repro import seed_everything


def resolve_metadata_dir(config: dict, artifact_root: Path) -> str:
    metadata_dir = Path(config["data"].get("metadata_dir", "dataset"))
    if not metadata_dir.is_absolute():
        metadata_dir = artifact_root / metadata_dir
    return str(metadata_dir.resolve())


def append_run_manifest(artifact_root: Path, run_dir: Path) -> None:
    metrics = load_json(run_dir / "metrics.json")
    best_dev = metrics.get("best_dev_metrics", {})
    manifest_row = {
        "run_name": metrics["run_name"],
        "training_time_seconds": metrics["training_time_seconds"],
        "peak_gpu_memory_gb": metrics["peak_gpu_memory_gb"],
        "total_parameters": metrics["total_parameters"],
        "trainable_parameters": metrics["trainable_parameters"],
        "best_epoch": best_dev.get("epoch"),
        "best_dev_mIoU": best_dev.get("mIoU"),
        "best_dev_mean_dice": best_dev.get("mean_dice"),
        "best_dev_hd95": best_dev.get("hd95"),
    }
    manifest_path = artifact_root / "run_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["run_name"] != metrics["run_name"]]
        manifest = pd.concat([manifest, pd.DataFrame([manifest_row])], ignore_index=True)
    else:
        manifest = pd.DataFrame([manifest_row])
    manifest.to_csv(manifest_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", required=True, help="VOC dataset root.")
    parser.add_argument("--artifact-root", default="./artifacts", help="Artifact output root.")
    parser.add_argument("--run-name", default=None, help="Optional run-name override.")
    parser.add_argument("--no-resume", action="store_true", help="Disable automatic resume from checkpoint_last.pth.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed_state = init_distributed()
    run_dir: Path | None = None
    try:
        config = load_config(args.config)
        if args.run_name is not None:
            config["experiment_name"] = args.run_name
        artifact_root = Path(args.artifact_root).resolve()
        config["data"]["metadata_dir"] = resolve_metadata_dir(config, artifact_root)
        run_dir = artifact_root / "runs" / config["experiment_name"]
        seed_everything(int(config["training"].get("seed", 42)), deterministic=bool(config["training"].get("deterministic", False)))

        run_dir = fit(
            config=config,
            data_root=args.data_root,
            artifact_root=artifact_root,
            distributed_state=distributed_state,
            resume=not args.no_resume,
        )
        if is_main_process(distributed_state):
            append_run_manifest(artifact_root, run_dir)
            print(f"Training complete. Artifacts saved to {run_dir}")
    except KeyboardInterrupt:
        if is_main_process(distributed_state) and run_dir is not None:
            progress_path = run_dir / "progress.json"
            progress = read_json_if_exists(progress_path)
            if progress is not None:
                progress["status"] = "interrupted"
                write_progress_file(progress_path, progress)
        raise
    finally:
        cleanup_distributed(distributed_state)


if __name__ == "__main__":
    main()
