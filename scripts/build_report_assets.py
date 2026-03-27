#!/usr/bin/env python3
"""Build tables, figures, and summary macros for the final report."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "artifacts" / "evals"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "artifacts" / "runs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "generated"

RUN_ORDER = [
    "unet_resnet34_pretrained_mps",
    "deeplabv3plus_resnet50_pretrained_mps",
    "segformer_b2_pretrained_mps",
    "sam2_hiera_s_frozen_pretrained_mps",
    "unet_resnet18_pretrained_mps",
    "unet_resnet50_pretrained_mps",
    "unet_loss_cross_entropy_pretrained_mps",
    "unet_loss_dice_pretrained_mps",
    "unet_aug_none_pretrained_mps",
    "unet_aug_strong_pretrained_mps",
    "sam2_hiera_s_lora_pretrained_mps",
]

HEADLINE_RUNS = [
    "unet_resnet34_pretrained_mps",
    "deeplabv3plus_resnet50_pretrained_mps",
    "segformer_b2_pretrained_mps",
    "sam2_hiera_s_lora_pretrained_mps",
]

DISPLAY_NAMES = {
    "unet_resnet18_pretrained_mps": "U-Net-R18",
    "unet_resnet34_pretrained_mps": "U-Net-R34",
    "unet_resnet50_pretrained_mps": "U-Net-R50",
    "deeplabv3plus_resnet50_pretrained_mps": "DeepLabV3+",
    "segformer_b2_pretrained_mps": "SegFormer-B2",
    "sam2_hiera_s_frozen_pretrained_mps": "SAM2-Frozen",
    "sam2_hiera_s_lora_pretrained_mps": "SAM2-LoRA",
    "unet_loss_cross_entropy_pretrained_mps": "U-Net CE",
    "unet_loss_dice_pretrained_mps": "U-Net Dice",
    "unet_aug_none_pretrained_mps": "U-Net NoAug",
    "unet_aug_strong_pretrained_mps": "U-Net StrongAug",
}

LONG_NAMES = {
    "unet_resnet18_pretrained_mps": "U-Net with ResNet-18 encoder",
    "unet_resnet34_pretrained_mps": "U-Net with ResNet-34 encoder",
    "unet_resnet50_pretrained_mps": "U-Net with ResNet-50 encoder",
    "deeplabv3plus_resnet50_pretrained_mps": "DeepLabV3+ with ResNet-50 backbone",
    "segformer_b2_pretrained_mps": "SegFormer-B2",
    "sam2_hiera_s_frozen_pretrained_mps": "SAM2 semantic adapter (frozen encoder)",
    "sam2_hiera_s_lora_pretrained_mps": "SAM2 semantic adapter with LoRA",
    "unet_loss_cross_entropy_pretrained_mps": "U-Net-ResNet34 with cross-entropy loss",
    "unet_loss_dice_pretrained_mps": "U-Net-ResNet34 with Dice loss",
    "unet_aug_none_pretrained_mps": "U-Net-ResNet34 without augmentation",
    "unet_aug_strong_pretrained_mps": "U-Net-ResNet34 with strong augmentation",
}

FAMILY_NAMES = {
    "unet_resnet18_pretrained_mps": "U-Net",
    "unet_resnet34_pretrained_mps": "U-Net",
    "unet_resnet50_pretrained_mps": "U-Net",
    "unet_loss_cross_entropy_pretrained_mps": "U-Net",
    "unet_loss_dice_pretrained_mps": "U-Net",
    "unet_aug_none_pretrained_mps": "U-Net",
    "unet_aug_strong_pretrained_mps": "U-Net",
    "deeplabv3plus_resnet50_pretrained_mps": "DeepLabV3+",
    "segformer_b2_pretrained_mps": "SegFormer",
    "sam2_hiera_s_frozen_pretrained_mps": "SAM2",
    "sam2_hiera_s_lora_pretrained_mps": "SAM2",
}

FAMILY_COLORS = {
    "U-Net": "#4c78a8",
    "DeepLabV3+": "#f58518",
    "SegFormer": "#54a24b",
    "SAM2": "#e45756",
}

ABLATION_GROUPS = [
    (
        "Backbone",
        "unet_resnet34_pretrained_mps",
        [
            "unet_resnet18_pretrained_mps",
            "unet_resnet34_pretrained_mps",
            "unet_resnet50_pretrained_mps",
        ],
    ),
    (
        "Loss",
        "unet_resnet34_pretrained_mps",
        [
            "unet_loss_cross_entropy_pretrained_mps",
            "unet_resnet34_pretrained_mps",
            "unet_loss_dice_pretrained_mps",
        ],
    ),
    (
        "Augmentation",
        "unet_resnet34_pretrained_mps",
        [
            "unet_aug_none_pretrained_mps",
            "unet_resnet34_pretrained_mps",
            "unet_aug_strong_pretrained_mps",
        ],
    ),
    (
        "SAM2 adaptation",
        "sam2_hiera_s_frozen_pretrained_mps",
        [
            "sam2_hiera_s_frozen_pretrained_mps",
            "sam2_hiera_s_lora_pretrained_mps",
        ],
    ),
]

SUBSET_ORDER = [
    "all",
    "person_present",
    "small_object",
    "crowded_scene",
    "high_boundary_complexity",
]

SUBSET_LABELS = {
    "all": "All",
    "person_present": "Person",
    "small_object": "Small Object",
    "crowded_scene": "Crowded",
    "high_boundary_complexity": "Boundary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", default="official_val")
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def metric_pct(value: float) -> float:
    if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
        return 100.0 * np.asarray(value, dtype=float)
    return 100.0 * float(value)


def read_experiment_bundle(run_name: str, eval_root: Path, run_root: Path, split: str) -> dict:
    eval_dir = eval_root / f"{run_name}_{split}"
    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing evaluation directory: {eval_dir}")

    run_dir = run_root / run_name
    metrics = load_json(eval_dir / "metrics.json")
    subset_metrics = load_json(eval_dir / "subset_metrics.json")
    per_class = pd.read_csv(eval_dir / "per_class.csv")
    per_image = pd.read_csv(eval_dir / "per_image.csv")
    train_metrics = load_json(run_dir / "metrics.json")
    train_log = pd.read_csv(run_dir / "train_log.csv")

    best_idx = int(train_log["dev_mIoU"].idxmax())
    best_row = train_log.iloc[best_idx]
    final_row = train_log.iloc[-1]

    row = {
        "run_name": run_name,
        "display_name": DISPLAY_NAMES[run_name],
        "long_name": LONG_NAMES[run_name],
        "family": FAMILY_NAMES[run_name],
        "mIoU": float(metrics["mIoU"]),
        "mean_dice": float(metrics["mean_dice"]),
        "hd95": float(metrics["hd95"]),
        "pixel_accuracy": float(metrics["pixel_accuracy"]),
        "images_per_second": float(metrics["images_per_second"]),
        "inference_time_seconds": float(metrics["inference_time_seconds"]),
        "training_time_seconds": float(train_metrics["training_time_seconds"]),
        "total_parameters": int(train_metrics["total_parameters"]),
        "trainable_parameters": int(train_metrics["trainable_parameters"]),
        "trainable_fraction": float(train_metrics["trainable_parameters"]) / float(train_metrics["total_parameters"]),
        "best_dev_mIoU": float(best_row["dev_mIoU"]),
        "best_dev_epoch": int(best_row["epoch"]),
        "final_epoch": int(final_row["epoch"]),
        "training_policy": train_metrics["training_policy"],
    }
    return {
        "summary": row,
        "metrics": metrics,
        "subset_metrics": subset_metrics,
        "per_class": per_class,
        "per_image": per_image,
        "train_log": train_log,
        "eval_dir": eval_dir,
        "run_dir": run_dir,
    }


def build_results_bundle(eval_root: Path, run_root: Path, split: str) -> dict[str, dict]:
    bundles = {}
    for run_name in RUN_ORDER:
        bundles[run_name] = read_experiment_bundle(run_name, eval_root, run_root, split)
    return bundles


def build_summary_frame(bundles: dict[str, dict]) -> pd.DataFrame:
    frame = pd.DataFrame([bundles[run_name]["summary"] for run_name in RUN_ORDER])
    frame["training_hours"] = frame["training_time_seconds"] / 3600.0
    frame["total_parameters_m"] = frame["total_parameters"] / 1_000_000.0
    frame["trainable_parameters_m"] = frame["trainable_parameters"] / 1_000_000.0
    return frame


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def maybe_bold(text: str, condition: bool) -> str:
    return rf"\textbf{{{text}}}" if condition else text


def format_percent(value: float, digits: int = 1) -> str:
    return f"{metric_pct(value):.{digits}f}"


def format_float(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def make_main_results_table(summary: pd.DataFrame, output_path: Path) -> None:
    table_df = summary[summary["run_name"].isin(HEADLINE_RUNS)].copy().sort_values("mIoU", ascending=False)
    best = {
        "mIoU": table_df["mIoU"].max(),
        "mean_dice": table_df["mean_dice"].max(),
        "pixel_accuracy": table_df["pixel_accuracy"].max(),
        "images_per_second": table_df["images_per_second"].max(),
    }
    best_hd95 = table_df["hd95"].min()

    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & mIoU & Dice & Pixel Acc. & HD95 $\downarrow$ & Train hrs & Infer FPS & Params (M) \\",
        r"\midrule",
    ]
    for row in table_df.itertuples():
        model = latex_escape(row.display_name)
        miou = maybe_bold(format_percent(row.mIoU), math.isclose(row.mIoU, best["mIoU"]))
        dice = maybe_bold(format_percent(row.mean_dice), math.isclose(row.mean_dice, best["mean_dice"]))
        pixel_acc = maybe_bold(format_percent(row.pixel_accuracy), math.isclose(row.pixel_accuracy, best["pixel_accuracy"]))
        hd95 = maybe_bold(format_float(row.hd95), math.isclose(row.hd95, best_hd95))
        train_hours = format_float(row.training_hours)
        fps = maybe_bold(format_float(row.images_per_second), math.isclose(row.images_per_second, best["images_per_second"]))
        params = format_float(row.total_parameters_m)
        lines.append(f"{model} & {miou} & {dice} & {pixel_acc} & {hd95} & {train_hours} & {fps} & {params} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(output_path, "\n".join(lines) + "\n")


def make_ablation_table(summary: pd.DataFrame, output_path: Path) -> None:
    indexed = summary.set_index("run_name")
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Group & Variant & mIoU & $\Delta$ mIoU & Dice & Train hrs \\",
        r"\midrule",
    ]
    for group_name, baseline, run_names in ABLATION_GROUPS:
        baseline_miou = float(indexed.loc[baseline, "mIoU"])
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{latex_escape(group_name)}}}}} \\")
        for run_name in run_names:
            row = indexed.loc[run_name]
            label = latex_escape(row["display_name"])
            delta = metric_pct(float(row["mIoU"]) - baseline_miou)
            delta_text = f"{delta:+.1f}"
            lines.append(
                f"{latex_escape(group_name)} & {label} & {format_percent(row['mIoU'])} & {delta_text} & "
                f"{format_percent(row['mean_dice'])} & {format_float(row['training_hours'])} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write_text(output_path, "\n".join(lines) + "\n")


def plot_runtime_vs_accuracy(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for family, frame in summary.groupby("family"):
        axis.scatter(
            frame["training_hours"],
            metric_pct(frame["mIoU"]),
            s=85,
            label=family,
            color=FAMILY_COLORS[family],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.4,
        )

    for row in summary.itertuples():
        axis.annotate(
            row.display_name,
            (row.training_hours, metric_pct(row.mIoU)),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    axis.set_xlabel("Training Time (hours)")
    axis.set_ylabel("Official val mIoU (%)")
    axis.set_title("Accuracy-Runtime Trade-off Across All Experiments")
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_per_class_heatmap(bundles: dict[str, dict], output_path: Path) -> None:
    matrices = []
    labels = []
    class_labels: list[str] | None = None
    for run_name in HEADLINE_RUNS:
        per_class = bundles[run_name]["per_class"].copy()
        per_class = per_class[per_class["class_id"] != 0]
        if class_labels is None:
            class_labels = per_class["class_name"].tolist()
        matrices.append(metric_pct(per_class["iou"].to_numpy()))
        labels.append(DISPLAY_NAMES[run_name])

    matrix = np.vstack(matrices)
    figure, axis = plt.subplots(figsize=(11.2, 3.6))
    heatmap = axis.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=95.0)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_xticks(np.arange(len(class_labels or [])))
    axis.set_xticklabels(class_labels or [], rotation=45, ha="right")
    axis.set_title("Per-Class IoU Heatmap for Headline Models")
    colorbar = figure.colorbar(heatmap, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("IoU (%)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_subset_heatmap(bundles: dict[str, dict], output_path: Path) -> None:
    selected_runs = [
        "unet_resnet34_pretrained_mps",
        "deeplabv3plus_resnet50_pretrained_mps",
        "segformer_b2_pretrained_mps",
        "sam2_hiera_s_frozen_pretrained_mps",
        "sam2_hiera_s_lora_pretrained_mps",
    ]
    matrix = []
    labels = []
    for run_name in selected_runs:
        subset_metrics = bundles[run_name]["subset_metrics"]
        matrix.append([metric_pct(subset_metrics[key]["mIoU"]) for key in SUBSET_ORDER])
        labels.append(DISPLAY_NAMES[run_name])

    array = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    heatmap = axis.imshow(array, cmap="YlOrRd", aspect="auto", vmin=float(array.min()) - 2.0, vmax=float(array.max()) + 2.0)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_xticks(np.arange(len(SUBSET_ORDER)))
    axis.set_xticklabels([SUBSET_LABELS[key] for key in SUBSET_ORDER], rotation=30, ha="right")
    axis.set_title("Subset Generalization mIoU on Official Validation")
    colorbar = figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("mIoU (%)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_unet_backbone_trajectories(bundles: dict[str, dict], output_path: Path) -> None:
    selected_runs = [
        "unet_resnet18_pretrained_mps",
        "unet_resnet34_pretrained_mps",
        "unet_resnet50_pretrained_mps",
    ]
    colors = {
        "unet_resnet18_pretrained_mps": "#9ecae9",
        "unet_resnet34_pretrained_mps": "#4c78a8",
        "unet_resnet50_pretrained_mps": "#1f3b73",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for run_name in selected_runs:
        log = bundles[run_name]["train_log"]
        axis.plot(
            log["epoch"],
            metric_pct(log["dev_mIoU"]),
            label=DISPLAY_NAMES[run_name],
            linewidth=2.0,
            color=colors[run_name],
        )
    axis.axvline(60, color="#666666", linestyle="--", linewidth=1.2, label="Old 60-epoch cap")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Internal dev mIoU (%)")
    axis.set_title("U-Net Backbone Ablation Trajectories")
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _copy_image_to_canvas(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), resample=Image.Resampling.BILINEAR)


def choose_headline_qualitative_ids(bundles: dict[str, dict]) -> list[str]:
    merged = None
    for run_name in HEADLINE_RUNS:
        frame = bundles[run_name]["per_image"][
            [
                "image_id",
                "person_present",
                "small_object",
                "crowded_scene",
                "high_boundary_complexity",
                "image_miou",
            ]
        ].rename(columns={"image_miou": f"{run_name}_miou"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame[["image_id", f"{run_name}_miou"]], on="image_id", how="inner")

    assert merged is not None
    merged["mean_miou"] = merged[[f"{run_name}_miou" for run_name in HEADLINE_RUNS]].mean(axis=1)
    merged["spread"] = merged[[f"{run_name}_miou" for run_name in HEADLINE_RUNS]].max(axis=1) - merged[
        [f"{run_name}_miou" for run_name in HEADLINE_RUNS]
    ].min(axis=1)

    selected_ids: list[str] = []

    crowded = merged[merged["crowded_scene"]].sort_values(["mean_miou", "spread"], ascending=[True, False])
    if not crowded.empty:
        selected_ids.append(str(crowded.iloc[0]["image_id"]))

    person_boundary = merged[(merged["person_present"]) | (merged["high_boundary_complexity"])]
    person_boundary = person_boundary[~person_boundary["image_id"].isin(selected_ids)]
    person_boundary = person_boundary.sort_values(["spread", "mean_miou"], ascending=[False, True])
    if not person_boundary.empty:
        selected_ids.append(str(person_boundary.iloc[0]["image_id"]))

    easy = merged[~merged["image_id"].isin(selected_ids)].sort_values(["mean_miou", "spread"], ascending=[False, False])
    if not easy.empty:
        selected_ids.append(str(easy.iloc[0]["image_id"]))

    return selected_ids


def build_headline_qualitative_grid(bundles: dict[str, dict], output_path: Path) -> dict[str, list[str]]:
    image_ids = choose_headline_qualitative_ids(bundles)
    if not image_ids:
        return {"image_ids": []}

    columns = ["Input", "Ground Truth"] + [DISPLAY_NAMES[run_name] for run_name in HEADLINE_RUNS]
    reference = bundles[HEADLINE_RUNS[0]]["per_image"].set_index("image_id")
    figure, axes = plt.subplots(len(image_ids), len(columns), figsize=(12.5, 2.85 * len(image_ids)))
    if len(image_ids) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, image_id in enumerate(image_ids):
        tags = []
        record = reference.loc[int(image_id)] if isinstance(reference.index[0], (int, np.integer)) else reference.loc[image_id]
        for key, label in [
            ("person_present", "person"),
            ("small_object", "small"),
            ("crowded_scene", "crowded"),
            ("high_boundary_complexity", "boundary"),
        ]:
            if bool(record[key]):
                tags.append(label)
        row_label = f"ID {image_id}"
        if tags:
            row_label += "\n" + ", ".join(tags)

        input_path = Path(record["input_path"])
        gt_path = Path(record["ground_truth_path"])
        row_images = [Image.open(input_path).convert("RGB"), Image.open(gt_path).convert("RGB")]
        for run_name in HEADLINE_RUNS:
            pred_path = Path(bundles[run_name]["eval_dir"] / "qualitative" / "predictions" / f"{image_id}.png")
            row_images.append(Image.open(pred_path).convert("RGB"))

        for col_idx, image in enumerate(row_images):
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(columns[col_idx], fontsize=9)
        axes[row_idx, 0].set_ylabel(row_label, rotation=0, labelpad=28, va="center", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"image_ids": image_ids}


def build_sam2_comparison_grid(bundles: dict[str, dict], output_path: Path) -> dict[str, list[str]]:
    frozen = bundles["sam2_hiera_s_frozen_pretrained_mps"]["per_image"][
        ["image_id", "image_miou", "person_present", "crowded_scene", "high_boundary_complexity", "input_path", "ground_truth_path"]
    ].rename(columns={"image_miou": "frozen_miou"})
    lora = bundles["sam2_hiera_s_lora_pretrained_mps"]["per_image"][["image_id", "image_miou"]].rename(
        columns={"image_miou": "lora_miou"}
    )
    merged = frozen.merge(lora, on="image_id", how="inner")
    merged["delta"] = merged["lora_miou"] - merged["frozen_miou"]
    candidates = merged[(merged["person_present"]) | (merged["crowded_scene"]) | (merged["high_boundary_complexity"])]
    selected = candidates.sort_values(["delta", "lora_miou"], ascending=[False, False]).head(2)
    if selected.empty:
        selected = merged.sort_values("delta", ascending=False).head(2)

    figure, axes = plt.subplots(len(selected), 4, figsize=(8.2, 2.9 * max(len(selected), 1)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in enumerate(selected.itertuples()):
        images = [
            Image.open(row.input_path).convert("RGB"),
            Image.open(row.ground_truth_path).convert("RGB"),
            Image.open(
                bundles["sam2_hiera_s_frozen_pretrained_mps"]["eval_dir"] / "qualitative" / "predictions" / f"{row.image_id}.png"
            ).convert("RGB"),
            Image.open(
                bundles["sam2_hiera_s_lora_pretrained_mps"]["eval_dir"] / "qualitative" / "predictions" / f"{row.image_id}.png"
            ).convert("RGB"),
        ]
        for col_idx, image in enumerate(images):
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(["Input", "Ground Truth", "SAM2-Frozen", "SAM2-LoRA"][col_idx], fontsize=9)
        axes[row_idx, 0].set_ylabel(f"ID {row.image_id}\nΔ={metric_pct(row.delta):+.1f}", rotation=0, labelpad=28, va="center", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"image_ids": [str(row.image_id) for row in selected.itertuples()]}


def build_summary_macros(summary: pd.DataFrame, output_path: Path) -> dict:
    ranked = summary.sort_values("mIoU", ascending=False).reset_index(drop=True)
    top = ranked.iloc[0]
    runner_up = ranked.iloc[1]
    segformer = summary[summary["run_name"] == "segformer_b2_pretrained_mps"].iloc[0]
    deeplab = summary[summary["run_name"] == "deeplabv3plus_resnet50_pretrained_mps"].iloc[0]
    sam2_frozen = summary[summary["run_name"] == "sam2_hiera_s_frozen_pretrained_mps"].iloc[0]
    sam2_lora = summary[summary["run_name"] == "sam2_hiera_s_lora_pretrained_mps"].iloc[0]
    unet34 = summary[summary["run_name"] == "unet_resnet34_pretrained_mps"].iloc[0]
    unet50 = summary[summary["run_name"] == "unet_resnet50_pretrained_mps"].iloc[0]
    strong_aug = summary[summary["run_name"] == "unet_aug_strong_pretrained_mps"].iloc[0]
    no_aug = summary[summary["run_name"] == "unet_aug_none_pretrained_mps"].iloc[0]

    macro_map = {
        "TopModelName": top["display_name"],
        "TopModelMiou": format_percent(top["mIoU"]),
        "RunnerUpName": runner_up["display_name"],
        "RunnerUpMiou": format_percent(runner_up["mIoU"]),
        "SegformerMiou": format_percent(segformer["mIoU"]),
        "SegformerVsDeeplabDelta": f"{metric_pct(segformer['mIoU'] - deeplab['mIoU']):.1f}",
        "SamLoRAGain": f"{metric_pct(sam2_lora['mIoU'] - sam2_frozen['mIoU']):.1f}",
        "UNetFiftyGain": f"{metric_pct(unet50['mIoU'] - unet34['mIoU']):.1f}",
        "StrongAugGain": f"{metric_pct(strong_aug['mIoU'] - no_aug['mIoU']):.1f}",
        "OfficialValCount": "213",
        "InternalTrainCount": "178",
        "InternalDevCount": "31",
        "CrowdedCount": "29",
        "PersonCount": "79",
        "SmallObjectCount": "117",
        "BoundaryCount": "56",
    }

    lines = []
    for key, value in macro_map.items():
        lines.append(rf"\newcommand{{\{key}}}{{{latex_escape(str(value))}}}")
    write_text(output_path, "\n".join(lines) + "\n")
    return macro_map


def build_report_summary(summary: pd.DataFrame, bundles: dict[str, dict], output_path: Path, qualitative_ids: dict, sam_ids: dict) -> None:
    ranked = summary.sort_values("mIoU", ascending=False)
    payload = {
        "ranked_runs": ranked[
            [
                "run_name",
                "display_name",
                "family",
                "mIoU",
                "mean_dice",
                "pixel_accuracy",
                "hd95",
                "training_hours",
            ]
        ].to_dict(orient="records"),
        "qualitative_ids": qualitative_ids,
        "sam2_comparison_ids": sam_ids,
    }
    save_json(output_path, payload)


def copy_reference_panels(bundles: dict[str, dict], figures_dir: Path) -> None:
    segformer_dir = bundles["segformer_b2_pretrained_mps"]["eval_dir"]
    shutil.copy2(segformer_dir / "best_worst_panel.png", figures_dir / "segformer_best_worst_panel.png")
    shutil.copy2(segformer_dir / "person_panel.png", figures_dir / "segformer_person_panel.png")


def main() -> None:
    args = parse_args()
    configure_plot_style()

    eval_root = Path(args.eval_root).resolve()
    run_root = Path(args.run_root).resolve()
    output_root = Path(args.output_root).resolve()
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundles = build_results_bundle(eval_root, run_root, args.split)
    summary = build_summary_frame(bundles)
    save_dataframe(summary.sort_values("mIoU", ascending=False), output_root / "official_val_summary.csv")

    make_main_results_table(summary, tables_dir / "main_results.tex")
    make_ablation_table(summary, tables_dir / "ablations.tex")

    plot_runtime_vs_accuracy(summary, figures_dir / "runtime_vs_accuracy.pdf")
    plot_per_class_heatmap(bundles, figures_dir / "headline_per_class_heatmap.pdf")
    plot_subset_heatmap(bundles, figures_dir / "subset_heatmap.pdf")
    plot_unet_backbone_trajectories(bundles, figures_dir / "unet_backbone_trajectories.pdf")

    qualitative_ids = build_headline_qualitative_grid(bundles, figures_dir / "headline_qualitative_grid.pdf")
    sam_ids = build_sam2_comparison_grid(bundles, figures_dir / "sam2_frozen_vs_lora.pdf")
    copy_reference_panels(bundles, figures_dir)
    build_summary_macros(summary, output_root / "report_macros.tex")
    build_report_summary(summary, bundles, output_root / "report_summary.json", qualitative_ids, sam_ids)
    print(f"Report assets saved to {output_root}")


if __name__ == "__main__":
    main()
