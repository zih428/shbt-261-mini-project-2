#!/usr/bin/env python3
"""Aggregate multiple evaluation directories into comparison tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from vocseg.utils.io import ensure_dir, load_json, save_dataframe
from vocseg.visualization.qualitative import plot_runtime_vs_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Evaluation directories to aggregate.")
    parser.add_argument("--output-dir", required=True, help="Output directory for aggregate artifacts.")
    return parser.parse_args()


def load_summary(run_dir: Path) -> dict:
    summary = load_json(run_dir / "metrics.json")
    summary["run_dir"] = str(run_dir)
    return summary


def build_main_results_table(run_dirs: list[Path]) -> pd.DataFrame:
    rows = [load_summary(run_dir) for run_dir in run_dirs]
    frame = pd.DataFrame(rows)
    return frame[
        [
            "run_name",
            "split",
            "mIoU",
            "mean_dice",
            "hd95",
            "pixel_accuracy",
            "training_time_seconds",
            "inference_time_seconds",
            "images_per_second",
            "total_parameters",
            "trainable_parameters",
        ]
    ].sort_values("mIoU", ascending=False)


def build_cross_model_mosaic(run_dirs: list[Path], output_path: Path, max_images: int = 4) -> None:
    per_image_frames = []
    for run_dir in run_dirs:
        frame = pd.read_csv(run_dir / "per_image.csv")
        frame["run_name"] = load_json(run_dir / "metrics.json")["run_name"]
        per_image_frames.append(frame[["image_id", "run_name", "image_miou"]])

    common_image_ids = sorted(set.intersection(*(set(frame["image_id"]) for frame in per_image_frames)))
    if not common_image_ids:
        return

    reference_run = run_dirs[0]
    reference_df = pd.read_csv(reference_run / "per_image.csv")
    image_ids = reference_df.sort_values("image_miou", ascending=False)["image_id"].head(max_images).tolist()

    figure, axes = plt.subplots(len(image_ids), len(run_dirs) + 2, figsize=(4 * (len(run_dirs) + 2), 4 * len(image_ids)))
    if len(image_ids) == 1:
        axes = [axes]

    for row_idx, image_id in enumerate(image_ids):
        input_image = Image.open(reference_run / "qualitative" / "inputs" / f"{image_id}.png").convert("RGB")
        gt_image = Image.open(reference_run / "qualitative" / "ground_truth" / f"{image_id}.png").convert("RGB")
        axes[row_idx][0].imshow(input_image)
        axes[row_idx][0].set_title(f"{image_id}\nInput")
        axes[row_idx][1].imshow(gt_image)
        axes[row_idx][1].set_title("Ground Truth")
        axes[row_idx][0].axis("off")
        axes[row_idx][1].axis("off")

        for col_idx, run_dir in enumerate(run_dirs, start=2):
            run_name = load_json(run_dir / "metrics.json")["run_name"]
            pred = Image.open(run_dir / "qualitative" / "predictions" / f"{image_id}.png").convert("RGB")
            axes[row_idx][col_idx].imshow(pred)
            axes[row_idx][col_idx].set_title(run_name)
            axes[row_idx][col_idx].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path).resolve() for path in args.run_dirs]
    output_dir = ensure_dir(args.output_dir)

    results = build_main_results_table(run_dirs)
    save_dataframe(results, output_dir / "main_results.csv")
    plot_runtime_vs_accuracy(results.dropna(subset=["training_time_seconds", "mIoU"]), output_dir / "runtime_vs_accuracy.png")
    build_cross_model_mosaic(run_dirs, output_dir / "cross_model_mosaic.png")
    print(f"Aggregate artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
