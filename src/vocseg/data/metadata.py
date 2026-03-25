"""Metadata extraction and split generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from vocseg.constants import IGNORE_INDEX, NUM_CLASSES, PERSON_CLASS_INDEX
from vocseg.utils.io import save_dataframe, save_json


def _mask_boundary(binary_mask: np.ndarray) -> np.ndarray:
    if not binary_mask.any():
        return np.zeros_like(binary_mask, dtype=bool)
    eroded = ndimage.binary_erosion(binary_mask, structure=np.ones((3, 3)), border_value=0)
    return binary_mask ^ eroded


def extract_mask_metadata(mask_array: np.ndarray) -> dict[str, Any]:
    valid = mask_array != IGNORE_INDEX
    foreground = (mask_array > 0) & valid
    present_classes = sorted(int(x) for x in np.unique(mask_array[foreground]))

    valid_pixels = int(valid.sum())
    foreground_pixels = int(foreground.sum())
    foreground_ratio = foreground_pixels / valid_pixels if valid_pixels else 0.0

    boundary = _mask_boundary(foreground)
    boundary_complexity = float(boundary.sum() / max(foreground_pixels, 1))

    object_areas: list[int] = []
    for class_id in present_classes:
        class_mask = mask_array == class_id
        labeled, num_features = ndimage.label(class_mask)
        if num_features == 0:
            continue
        counts = np.bincount(labeled.ravel())[1:]
        object_areas.extend(int(x) for x in counts if x > 0)

    median_object_area = float(np.median(object_areas)) if object_areas else math.nan
    return {
        "classes_present": present_classes,
        "person_present": PERSON_CLASS_INDEX in present_classes,
        "num_foreground_classes": len(present_classes),
        "foreground_area_ratio": foreground_ratio,
        "boundary_complexity": boundary_complexity,
        "median_object_area": median_object_area,
        "num_instances": len(object_areas),
    }


def build_metadata_for_dataset(dataset: Any, official_split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    images = getattr(dataset, "images", None)

    for dataset_index in range(len(dataset)):
        image, mask = dataset[dataset_index]
        image_array = np.array(image)
        mask_array = np.array(mask, dtype=np.int64)
        image_path = Path(images[dataset_index]) if images is not None else Path(f"{official_split}_{dataset_index}.jpg")
        stats = extract_mask_metadata(mask_array)
        rows.append(
            {
                "official_split": official_split,
                "dataset_index": dataset_index,
                "image_id": image_path.stem,
                "width": int(image_array.shape[1]),
                "height": int(image_array.shape[0]),
                **stats,
            }
        )

    frame = pd.DataFrame(rows)
    frame["classes_present"] = frame["classes_present"].apply(list)
    frame["crowded_scene"] = frame["num_foreground_classes"] >= 3
    return frame


def build_class_presence_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((len(frame), NUM_CLASSES - 1), dtype=np.int64)
    for row_index, classes_present in enumerate(frame["classes_present"]):
        for class_id in classes_present:
            if class_id == 0 or class_id == IGNORE_INDEX:
                continue
            matrix[row_index, class_id - 1] = 1
    return matrix


def iterative_multilabel_split(label_matrix: np.ndarray, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic label-aware split heuristic.

    This is not a third-party implementation of Sechidis et al.; it approximates iterative
    multilabel stratification while staying dependency-light and deterministic.
    """

    rng = np.random.default_rng(seed)
    num_samples = label_matrix.shape[0]
    target_val = int(round(num_samples * val_fraction))
    target_train = num_samples - target_val

    label_freq = label_matrix.sum(axis=0).astype(np.float64)
    rarity = []
    shuffled = rng.permutation(num_samples)
    for index in range(num_samples):
        labels = label_matrix[index] > 0
        if labels.any():
            score = float((1.0 / np.maximum(label_freq[labels], 1.0)).sum())
        else:
            score = 0.0
        rarity.append(score)

    order = sorted(range(num_samples), key=lambda idx: (-rarity[idx], int(np.where(shuffled == idx)[0][0])))

    desired_val = label_freq * val_fraction
    desired_train = label_freq - desired_val
    current_val = np.zeros_like(label_freq)
    current_train = np.zeros_like(label_freq)

    val_indices: list[int] = []
    train_indices: list[int] = []

    for index in order:
        sample_labels = label_matrix[index] > 0
        if len(val_indices) >= target_val:
            assign_val = False
        elif len(train_indices) >= target_train:
            assign_val = True
        else:
            gain_val = float(np.clip(desired_val[sample_labels] - current_val[sample_labels], a_min=0.0, a_max=None).sum())
            gain_train = float(np.clip(desired_train[sample_labels] - current_train[sample_labels], a_min=0.0, a_max=None).sum())

            if gain_val > gain_train:
                assign_val = True
            elif gain_train > gain_val:
                assign_val = False
            else:
                val_fill = len(val_indices) / max(target_val, 1)
                train_fill = len(train_indices) / max(target_train, 1)
                if val_fill < train_fill:
                    assign_val = True
                elif train_fill < val_fill:
                    assign_val = False
                else:
                    assign_val = bool(rng.integers(0, 2))

        if assign_val:
            val_indices.append(index)
            current_val += label_matrix[index]
        else:
            train_indices.append(index)
            current_train += label_matrix[index]

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def finalize_metadata(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train_frame = train_frame.copy()
    val_frame = val_frame.copy()

    valid_train_medians = train_frame["median_object_area"].dropna()
    median_object_threshold = float(valid_train_medians.median()) if not valid_train_medians.empty else math.nan
    boundary_threshold = float(train_frame["boundary_complexity"].quantile(0.75))

    thresholds = {
        "small_object_threshold": median_object_threshold,
        "high_boundary_complexity_threshold": boundary_threshold,
    }

    for frame in (train_frame, val_frame):
        frame["small_object"] = False
        if not math.isnan(median_object_threshold):
            frame["small_object"] = frame["median_object_area"].fillna(np.inf) <= median_object_threshold
        frame["high_boundary_complexity"] = frame["boundary_complexity"] >= boundary_threshold
        frame["subset_tags"] = frame.apply(
            lambda row: [
                tag
                for tag, condition in [
                    ("person_present", bool(row["person_present"])),
                    ("small_object", bool(row["small_object"])),
                    ("crowded_scene", bool(row["crowded_scene"])),
                    ("high_boundary_complexity", bool(row["high_boundary_complexity"])),
                ]
                if condition
            ],
            axis=1,
        )

    return train_frame, val_frame, thresholds


def save_metadata_artifacts(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    train_indices: list[int],
    dev_indices: list[int],
    thresholds: dict[str, float],
    output_dir: str | Path,
    seed: int,
    val_fraction: float,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_serializable = train_frame.copy()
    val_serializable = val_frame.copy()
    for frame in (train_serializable, val_serializable):
        frame["classes_present"] = frame["classes_present"].apply(lambda values: ",".join(str(x) for x in values))
        frame["subset_tags"] = frame["subset_tags"].apply(lambda values: ",".join(values))

    save_dataframe(train_serializable, output_dir / "train_metadata.csv")
    save_dataframe(val_serializable, output_dir / "val_metadata.csv")
    manifest = {
        "seed": seed,
        "val_fraction": val_fraction,
        "internal_train": train_indices,
        "internal_dev": dev_indices,
        "official_val": list(range(len(val_frame))),
        "thresholds": thresholds,
    }
    save_json(manifest, output_dir / "split_manifest.json")
