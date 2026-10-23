from pathlib import Path

import numpy as np
import pytest

from autocar.ml.dataset import (
    load_4d_training_data,
    load_6d_training_data,
    train_validation_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_4D_PATH = PROJECT_ROOT / "data" / "train4dAll.txt"
TRAIN_6D_PATH = PROJECT_ROOT / "data" / "train6dAll.txt"


def test_load_4d_training_data():
    features, target = load_4d_training_data(TRAIN_4D_PATH)

    assert features.shape == (1475, 3)
    assert target.shape == (1475, 1)


def test_load_6d_training_data():
    features, target = load_6d_training_data(TRAIN_6D_PATH)

    assert features.shape == (1475, 5)
    assert target.shape == (1475, 1)


def test_train_validation_split_shapes():
    features, target = load_4d_training_data(TRAIN_4D_PATH)

    train_x, train_y, validation_x, validation_y = (
        train_validation_split(features, target)
    )

    assert train_x.shape == (1180, 3)
    assert train_y.shape == (1180, 1)
    assert validation_x.shape == (295, 3)
    assert validation_y.shape == (295, 1)


def test_same_seed_produces_same_split():
    features, target = load_4d_training_data(TRAIN_4D_PATH)

    first_split = train_validation_split(features, target, seed=42)
    second_split = train_validation_split(features, target, seed=42)

    for first, second in zip(first_split, second_split):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.1])
def test_invalid_validation_ratio_raises_error(ratio):
    features = np.zeros((10, 3))
    target = np.zeros((10, 1))

    with pytest.raises(ValueError):
        train_validation_split(
            features,
            target,
            validation_ratio=ratio,
        )


def test_mismatched_sample_count_raises_error():
    features = np.zeros((10, 3))
    target = np.zeros((9, 1))

    with pytest.raises(ValueError):
        train_validation_split(features, target)
