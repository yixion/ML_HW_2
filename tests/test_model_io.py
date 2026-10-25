import numpy as np
import pytest

from autocar.ml.mlp import MLPRegressor
from autocar.ml.preprocessing import StandardScaler
from autocar.storage.model_io import (
    load_model,
    save_model,
)


def test_save_and_load_model_preserves_predictions(
    tmp_path,
):
    features = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
    ])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = MLPRegressor(
        input_size=3,
        hidden_sizes=(4, 2),
        seed=42,
    )
    model.l_r = 0.05

    expected_predictions = model.predict(
        scaled_features
    )

    model_path = tmp_path / "model.npz"

    save_model(
        model_path,
        model,
        scaler,
    )

    loaded_model, loaded_scaler = load_model(
        model_path
    )

    loaded_scaled_features = (
        loaded_scaler.transform(features)
    )
    actual_predictions = loaded_model.predict(
        loaded_scaled_features
    )

    assert model_path.exists()
    assert loaded_model.layer_sizes == model.layer_sizes
    assert loaded_model.l_r == pytest.approx(model.l_r)

    for original_weight, loaded_weight in zip(
        model.W,
        loaded_model.W,
    ):
        np.testing.assert_allclose(
            loaded_weight,
            original_weight,
        )

    np.testing.assert_allclose(
        loaded_scaled_features,
        scaled_features,
    )
    np.testing.assert_allclose(
        actual_predictions,
        expected_predictions,
    )
