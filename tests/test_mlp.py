import numpy as np
import pytest

from autocar.ml.mlp import MLPRegressor


def test_parameter_shapes_with_tuple_hidden_sizes():
    model = MLPRegressor(
        input_size=3,
        hidden_sizes=(4, 2),
        seed=42,
    )

    assert len(model.W) == 3

    # 每一層輸入多一列 bias weight
    assert model.W[0].shape == (4, 4)
    assert model.W[1].shape == (5, 2)
    assert model.W[2].shape == (3, 1)


def test_accepts_numpy_array_hidden_sizes():
    model = MLPRegressor(
        input_size=3,
        hidden_sizes=np.array([4, 2]),
        seed=42,
    )

    assert model.hidden_sizes == (4, 2)
    assert model.layer_sizes == (3, 4, 2, 1)

    assert model.W[0].shape == (4, 4)
    assert model.W[1].shape == (5, 2)
    assert model.W[2].shape == (3, 1)


def test_bias_weights_start_at_zero():
    model = MLPRegressor(
        input_size=3,
        hidden_sizes=(4, 2),
        seed=42,
    )

    for weight_matrix in model.W:
        bias_weights = weight_matrix[0, :]

        np.testing.assert_array_equal(
            bias_weights,
            np.zeros_like(bias_weights),
        )

@pytest.mark.skip(
    reason="Xavier initialization  尚未實作"
)

def test_non_bias_weights_follow_xavier_bounds():
    model = MLPRegressor(
        input_size=3,
        hidden_sizes=(4, 2),
        seed=42,
    )

    for layer_index in range(
        1,
        len(model.layer_sizes),
    ):
        input_count = model.layer_sizes[layer_index - 1]
        output_count = model.layer_sizes[layer_index]

        limit = np.sqrt(
            6 / (input_count + output_count)
        )

        non_bias_weights = model.W[layer_index - 1][
            :-1,
            :
        ]

        assert np.all(
            np.abs(non_bias_weights) <= limit
        )


def test_same_seed_produces_same_parameters():
    first = MLPRegressor(
        input_size=3,
        hidden_sizes=(4, 2),
        seed=42,
    )
    second = MLPRegressor(
        input_size=3,
        hidden_sizes=(4, 2),
        seed=42,
    )

    for first_weight, second_weight in zip(
        first.W,
        second.W,
    ):
        np.testing.assert_allclose(
            first_weight,
            second_weight,
        )


def test_forward_output_shape():
    model = MLPRegressor(
        input_size=3,
        hidden_sizes=(4, 2),
        seed=42,
    )
    features = np.zeros((10, 3))

    predictions = model.forward(features)

    assert predictions.shape == (10, 1)
    assert np.isfinite(predictions).all()


def test_forward_computes_expected_values():
    model = MLPRegressor(
        input_size=1,
        hidden_sizes=(1,),
        seed=42,
    )

    # 第一列是一般 weight，最後一列是 bias。
    model.W[0] = np.array([
        [0.0],
        [1.0],
    ])
    model.W[1] = np.array([
        [-0.5],
        [2.0],
    ])

    features = np.array([
        [0.0],
        [1.0],
    ])

    predictions = model.forward(features)

    expected = 2.0 * (1/(1+np.exp((-1.0)*features))) + 0.5

    np.testing.assert_allclose(
        predictions,
        expected,
    )


@pytest.mark.parametrize(
    "invalid_features",
    [
        np.array([1.0, 2.0, 3.0]),
        np.zeros((2, 3, 1)),
    ],
)
def test_forward_rejects_non_2d_features(
    invalid_features,
):
    model = MLPRegressor(
        input_size=3,
        hidden_sizes=(4, 2),
        seed=42,
    )

    with pytest.raises(ValueError):
        model.forward(invalid_features)


def test_forward_rejects_wrong_feature_count():
    model = MLPRegressor(
        input_size=3,
        hidden_sizes=(4, 2),
        seed=42,
    )
    invalid_features = np.zeros((10, 5))

    with pytest.raises(ValueError):
        model.forward(invalid_features)


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf],
)
def test_forward_rejects_non_finite_values(
    invalid_value,
):
    model = MLPRegressor(
        input_size=3,
        hidden_sizes=(4, 2),
        seed=42,
    )
    features = np.zeros((2, 3))
    features[0, 0] = invalid_value

    with pytest.raises(ValueError):
        model.forward(features)
