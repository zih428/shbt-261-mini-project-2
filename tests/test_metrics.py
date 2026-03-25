import math

import numpy as np

from vocseg.evaluation.metrics import confusion_matrix_from_arrays, hd95_multiclass, metrics_from_confusion


def test_perfect_prediction_metrics():
    target = np.array([[0, 1], [1, 2]], dtype=np.int64)
    prediction = target.copy()
    matrix = confusion_matrix_from_arrays(prediction, target, num_classes=3, ignore_index=255)
    summary, per_class = metrics_from_confusion(matrix, class_names=["bg", "a", "b"])
    assert math.isclose(summary["mIoU"], 1.0)
    assert math.isclose(summary["mean_dice"], 1.0)
    assert math.isclose(summary["pixel_accuracy"], 1.0)
    assert per_class["iou"].dropna().eq(1.0).all()


def test_hd95_zero_for_perfect_match():
    target = np.array([[0, 1], [1, 1]], dtype=np.int64)
    prediction = target.copy()
    score, by_class = hd95_multiclass(prediction, target, num_classes=2, include_background=False)
    assert math.isclose(score, 0.0)
    assert all(math.isclose(value, 0.0) for value in by_class.values())
