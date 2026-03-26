#!/usr/bin/env python3
"""Evaluate a trained segmentation checkpoint and save full analysis artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from vocseg.config import dump_config, load_config
from vocseg.constants import NUM_CLASSES
from vocseg.data import build_dataset_bundle, segmentation_collate_fn
from vocseg.models import build_model
from vocseg.training.engine import evaluate_model
from vocseg.utils.io import ensure_dir, load_json, save_dataframe, save_json
from vocseg.visualization.qualitative import plot_best_worst_triptychs, plot_per_class_iou, plot_person_panel


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_metadata_dir(config: dict, artifact_root: Path) -> str:
    metadata_dir = Path(config["data"].get("metadata_dir", "dataset"))
    if not metadata_dir.is_absolute():
        metadata_dir = artifact_root / metadata_dir
    return str(metadata_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--data-root", required=True, help="VOC dataset root.")
    parser.add_argument("--artifact-root", default="./artifacts", help="Artifact output root.")
    parser.add_argument("--split", default="official_val", choices=["internal_train", "internal_dev", "official_val"])
    parser.add_argument("--batch-size", type=int, default=None, help="Optional eval batch-size override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifact_root = Path(args.artifact_root).resolve()
    config["data"]["metadata_dir"] = resolve_metadata_dir(config, artifact_root)

    dataset_bundle = build_dataset_bundle(config["data"], data_root=args.data_root, metadata_dir=config["data"]["metadata_dir"])
    eval_dataset = dataset_bundle[args.split]
    eval_batch_size = args.batch_size or int(config["data"].get("eval_batch_size", config["data"]["batch_size"]))
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=segmentation_collate_fn,
    )

    device = resolve_device()
    model = build_model(config["model"], num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    experiment_name = config["experiment_name"]
    eval_dir = ensure_dir(artifact_root / "evals" / f"{experiment_name}_{args.split}")
    dump_config(config, eval_dir / "resolved_config.yaml")

    results = evaluate_model(
        model=model,
        loader=eval_loader,
        device=device,
        save_dir=eval_dir,
        save_qualitative_assets=True,
        include_background_hd95=bool(config["metrics"].get("hd95_include_background", False)),
    )
    summary = results["summary"]
    per_class = results["per_class"]
    per_image = results["per_image"]
    subset_metrics = results["subset_metrics"]

    train_metrics_path = Path(args.checkpoint).parent / "metrics.json"
    if train_metrics_path.exists():
        train_metrics = load_json(train_metrics_path)
        summary["training_time_seconds"] = train_metrics.get("training_time_seconds")
        summary["peak_gpu_memory_gb"] = train_metrics.get("peak_gpu_memory_gb")
        summary["total_parameters"] = train_metrics.get("total_parameters")
        summary["trainable_parameters"] = train_metrics.get("trainable_parameters")

    summary["run_name"] = experiment_name
    summary["split"] = args.split
    save_json(summary, eval_dir / "metrics.json")
    save_dataframe(per_class, eval_dir / "per_class.csv")
    save_dataframe(per_image, eval_dir / "per_image.csv")
    save_json(subset_metrics, eval_dir / "subset_metrics.json")

    plot_per_class_iou(per_class, eval_dir / "per_class_iou.png")
    plot_best_worst_triptychs(per_image, eval_dir, eval_dir / "best_worst_panel.png", top_k=3)
    plot_person_panel(per_image, eval_dir, eval_dir / "person_panel.png", count=6)
    print(f"Evaluation complete. Artifacts saved to {eval_dir}")


if __name__ == "__main__":
    main()
