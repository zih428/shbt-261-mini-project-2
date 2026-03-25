"""Segmentation metrics, including HD95 and subset analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from vocseg.constants import IGNORE_INDEX, NUM_CLASSES, VOC_CLASSES


def confusion_matrix_from_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    valid = target != ignore_index
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    prediction = prediction[valid].astype(np.int64)
    target = target[valid].astype(np.int64)
    encoded = target * num_classes + prediction
    matrix = np.bincount(encoded, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def metrics_from_confusion(matrix: np.ndarray, class_names: list[str] | None = None) -> tuple[dict[str, float], pd.DataFrame]:
    class_names = class_names or VOC_CLASSES
    true_positive = np.diag(matrix).astype(np.float64)
    gt_pixels = matrix.sum(axis=1).astype(np.float64)
    pred_pixels = matrix.sum(axis=0).astype(np.float64)
    union = gt_pixels + pred_pixels - true_positive
    dice_denom = gt_pixels + pred_pixels

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
        dice = np.divide(2.0 * true_positive, dice_denom, out=np.full_like(true_positive, np.nan), where=dice_denom > 0)
        accuracy = np.divide(true_positive, gt_pixels, out=np.full_like(true_positive, np.nan), where=gt_pixels > 0)

    total = matrix.sum()
    pixel_accuracy = float(true_positive.sum() / total) if total > 0 else math.nan
    summary = {
        "mIoU": float(np.nanmean(iou)),
        "mean_dice": float(np.nanmean(dice)),
        "pixel_accuracy": pixel_accuracy,
        "mean_class_accuracy": float(np.nanmean(accuracy)),
    }
    per_class = pd.DataFrame(
        {
            "class_id": list(range(len(class_names))),
            "class_name": class_names,
            "iou": iou,
            "dice": dice,
            "accuracy": accuracy,
            "gt_pixels": gt_pixels,
            "pred_pixels": pred_pixels,
            "tp_pixels": true_positive,
        }
    )
    return summary, per_class


def _surface_points(binary_mask: np.ndarray) -> np.ndarray:
    if not binary_mask.any():
        return np.empty((0, 2), dtype=np.float32)
    eroded = ndimage.binary_erosion(binary_mask, structure=np.ones((3, 3)), border_value=0)
    boundary = binary_mask ^ eroded
    points = np.argwhere(boundary)
    if points.size == 0:
        points = np.argwhere(binary_mask)
    return points.astype(np.float32)


def hd95_binary(prediction: np.ndarray, target: np.ndarray, fallback_distance: float) -> float | None:
    if not target.any():
        return None
    if not prediction.any():
        return fallback_distance

    pred_points = _surface_points(prediction)
    gt_points = _surface_points(target)
    if pred_points.size == 0 or gt_points.size == 0:
        return fallback_distance

    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    d_gt_to_pred = pred_tree.query(gt_points, k=1)[0]
    d_pred_to_gt = gt_tree.query(pred_points, k=1)[0]
    return float(max(np.percentile(d_gt_to_pred, 95), np.percentile(d_pred_to_gt, 95)))


def hd95_multiclass(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
    include_background: bool = False,
) -> tuple[float, dict[int, float]]:
    valid = target != ignore_index
    prediction = np.where(valid, prediction, 0)
    target = np.where(valid, target, 0)

    class_range = range(0 if include_background else 1, num_classes)
    diagonal = float(math.sqrt(target.shape[0] ** 2 + target.shape[1] ** 2))

    scores: dict[int, float] = {}
    for class_id in class_range:
        pred_mask = prediction == class_id
        target_mask = target == class_id
        score = hd95_binary(pred_mask, target_mask, fallback_distance=diagonal)
        if score is not None:
            scores[class_id] = score
    mean_hd95 = float(np.mean(list(scores.values()))) if scores else math.nan
    return mean_hd95, scores


@dataclass
class SegmentationMetricAccumulator:
    num_classes: int = NUM_CLASSES
    class_names: list[str] = field(default_factory=lambda: VOC_CLASSES)
    ignore_index: int = IGNORE_INDEX
    include_background_hd95: bool = False
    records: list[dict[str, Any]] = field(default_factory=list)
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
    hd95_by_class: dict[int, list[float]] = field(default_factory=dict)

    def update(self, prediction: np.ndarray, target: np.ndarray, meta: dict[str, Any]) -> None:
        sample_confusion = confusion_matrix_from_arrays(prediction, target, self.num_classes, self.ignore_index)
        self.confusion += sample_confusion

        sample_summary, _ = metrics_from_confusion(sample_confusion, class_names=self.class_names)
        sample_hd95, sample_hd95_per_class = hd95_multiclass(
            prediction,
            target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            include_background=self.include_background_hd95,
        )

        for class_id, score in sample_hd95_per_class.items():
            self.hd95_by_class.setdefault(class_id, []).append(score)

        record = {
            "image_id": meta["image_id"],
            "official_split": meta["official_split"],
            "dataset_index": meta["dataset_index"],
            "image_miou": sample_summary["mIoU"],
            "image_dice": sample_summary["mean_dice"],
            "image_pixel_accuracy": sample_summary["pixel_accuracy"],
            "image_hd95": sample_hd95,
            "person_present": bool(meta.get("person_present", False)),
            "small_object": bool(meta.get("small_object", False)),
            "crowded_scene": bool(meta.get("crowded_scene", False)),
            "high_boundary_complexity": bool(meta.get("high_boundary_complexity", False)),
            "num_foreground_classes": int(meta.get("num_foreground_classes", 0)),
            "_confusion": sample_confusion,
            "_hd95_by_class": sample_hd95_per_class,
        }
        self.records.append(record)

    def finalize(self) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
        overall_summary, per_class = metrics_from_confusion(self.confusion, class_names=self.class_names)
        overall_hd95 = [score for scores in self.hd95_by_class.values() for score in scores]
        overall_summary["hd95"] = float(np.mean(overall_hd95)) if overall_hd95 else math.nan
        overall_summary["num_images"] = len(self.records)

        per_class = per_class.copy()
        per_class["hd95"] = per_class["class_id"].map(
            lambda class_id: float(np.mean(self.hd95_by_class[class_id])) if class_id in self.hd95_by_class else math.nan
        )

        per_image_rows = []
        for record in self.records:
            row = {key: value for key, value in record.items() if not key.startswith("_")}
            per_image_rows.append(row)
        per_image = pd.DataFrame(per_image_rows)

        subset_metrics = self._build_subset_metrics()
        return overall_summary, per_class, per_image, subset_metrics

    def _build_subset_metrics(self) -> dict[str, dict[str, Any]]:
        subsets = {
            "all": self.records,
            "person_present": [row for row in self.records if row["person_present"]],
            "small_object": [row for row in self.records if row["small_object"]],
            "crowded_scene": [row for row in self.records if row["crowded_scene"]],
            "high_boundary_complexity": [row for row in self.records if row["high_boundary_complexity"]],
        }
        outputs: dict[str, dict[str, Any]] = {}
        for subset_name, rows in subsets.items():
            if not rows:
                continue
            matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            hd95_values: list[float] = []
            for row in rows:
                matrix += row["_confusion"]
                hd95_values.extend(row["_hd95_by_class"].values())
            summary, _ = metrics_from_confusion(matrix, class_names=self.class_names)
            summary["hd95"] = float(np.mean(hd95_values)) if hd95_values else math.nan
            summary["num_images"] = len(rows)
            outputs[subset_name] = summary
        return outputs
