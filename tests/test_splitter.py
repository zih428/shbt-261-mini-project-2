import numpy as np

from vocseg.data.metadata import iterative_multilabel_split


def test_iterative_split_is_deterministic():
    labels = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
        ]
    )
    train_a, val_a = iterative_multilabel_split(labels, val_fraction=0.33, seed=42)
    train_b, val_b = iterative_multilabel_split(labels, val_fraction=0.33, seed=42)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) + len(val_a) == len(labels)
