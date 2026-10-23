import numpy as np
import pytest

from autocar.ml.preprocessing import StandardScaler


def test_fit_stores_mean_and_scale():
    features = np.array([
        [1.0, 10.0],
        [3.0, 30.0],
    ])
    scaler = StandardScaler()

    scaler.fit(features)

    np.testing.assert_allclose(
        scaler.mean_,
        np.array([2.0, 20.0]),
    )
    np.testing.assert_allclose(
        scaler.scale_,
        np.array([1.0, 10.0]),
    )


def test_transform_standardizes_features():
    features = np.array([
        [1.0, 10.0],
        [3.0, 30.0],
    ])
    expected = np.array([
        [-1.0, -1.0],
        [1.0, 1.0],
    ])
    scaler = StandardScaler()

    scaler.fit(features)
    transformed = scaler.transform(features)

    np.testing.assert_allclose(transformed, expected)


def test_fit_transform_matches_fit_then_transform():
    features = np.array([
        [1.0, 10.0],
        [3.0, 30.0],
        [5.0, 50.0],
    ])

    first_scaler = StandardScaler()
    first_scaler.fit(features)
    expected = first_scaler.transform(features)

    second_scaler = StandardScaler()
    actual = second_scaler.fit_transform(features)

    np.testing.assert_allclose(actual, expected)


def test_inverse_transform_restores_original_features():
    features = np.array([
        [1.0, 10.0],
        [3.0, 30.0],
        [5.0, 50.0],
    ])
    scaler = StandardScaler()

    transformed = scaler.fit_transform(features)
    restored = scaler.inverse_transform(transformed)

    np.testing.assert_allclose(restored, features)


def test_constant_column_does_not_produce_invalid_values():
    features = np.array([
        [1.0, 10.0],
        [3.0, 10.0],
    ])
    scaler = StandardScaler()

    transformed = scaler.fit_transform(features)

    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(
        transformed[:, 1],
        np.array([0.0, 0.0]),
    )


def test_transform_before_fit_raises_runtime_error():
    features = np.array([
        [1.0, 10.0],
        [3.0, 30.0],
    ])
    scaler = StandardScaler()

    with pytest.raises(RuntimeError):
        scaler.transform(features)


def test_transform_rejects_different_feature_count():
    training_features = np.array([
        [1.0, 10.0],
        [3.0, 30.0],
    ])
    invalid_features = np.array([
        [1.0, 10.0, 100.0],
    ])
    scaler = StandardScaler()
    scaler.fit(training_features)

    with pytest.raises(ValueError):
        scaler.transform(invalid_features)
