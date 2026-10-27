import numpy as np
import pytest

from autocar.simulation.simulator import Simulator
from autocar.simulation.track import Track
from autocar.simulation.vehicle import Vehicle


class RecordingScaler:
    def __init__(self):
        self.received_features = None

    def transform(self, features):
        self.received_features = features.copy()
        return features


class ConstantModel:
    def __init__(
        self,
        input_size,
        prediction=0.0,
    ):
        self.layer_sizes = (input_size, 1)
        self.prediction = prediction
        self.received_features = None

    def predict(self, features):
        self.received_features = features.copy()

        return np.array([
            [self.prediction],
        ])


def create_test_track():
    return Track(
        start_position=np.array([
            0.0,
            0.0,
            90.0,
        ]),
        goal_top_left=np.array([
            -1.0,
            20.0,
        ]),
        goal_bottom_right=np.array([
            1.0,
            18.0,
        ]),
        boundary=np.array([
            [-5.0, -5.0],
            [5.0, -5.0],
            [5.0, 10.0],
            [-5.0, 10.0],
            [-5.0, -5.0],
        ]),
    )
@pytest.mark.parametrize(
    "input_size, expected_features",
    [
        (
            3,
            np.array([[
                10.0,
                np.sqrt(50.0),
                np.sqrt(50.0),
            ]]),
        ),
        (
            5,
            np.array([[
                0.0,
                0.0,
                10.0,
                np.sqrt(50.0),
                np.sqrt(50.0),
            ]]),
        ),
    ],
)
def test_simulator_step_builds_expected_features(
    input_size,
    expected_features,
):
    track = create_test_track()
    vehicle = Vehicle(
        x=0.0,
        y=0.0,
        heading_degrees=90.0,
    )
    scaler = RecordingScaler()
    model = ConstantModel(
        input_size=input_size,
        prediction=0.0,
    )

    simulator = Simulator(
        track=track,
        vehicle=vehicle,
        model=model,
        scaler=scaler,
    )

    status = simulator.step()

    assert status == "running"

    np.testing.assert_allclose(
        scaler.received_features,
        expected_features,
    )
    np.testing.assert_allclose(
        model.received_features,
        expected_features,
    )

    # steering 為 0，因此車輛向上移動一單位
    assert vehicle.x == pytest.approx(0.0)
    assert vehicle.y == pytest.approx(1.0)
    assert vehicle.heading_degrees == pytest.approx(
        90.0
    )

    assert len(simulator.records) == 1

    record = simulator.records[0]

    # record 必須保存移動前的位置
    assert record.x == pytest.approx(0.0)
    assert record.y == pytest.approx(0.0)
    assert record.front_distance == pytest.approx(
        10.0
    )
    assert record.right_distance == pytest.approx(
        np.sqrt(50.0)
    )
    assert record.left_distance == pytest.approx(
        np.sqrt(50.0)
    )
    assert record.steering_degrees == pytest.approx(
        0.0
    )
    assert np.isscalar(vehicle.x)
    assert np.isscalar(vehicle.y)
    assert np.isscalar(
        vehicle.heading_degrees
    )

    assert np.isscalar(
        simulator.records[0].steering_degrees
    )
def test_simulator_step_clips_steering():
    track = create_test_track()
    vehicle = Vehicle(
        x=0.0,
        y=0.0,
        heading_degrees=90.0,
    )
    scaler = RecordingScaler()
    model = ConstantModel(
        input_size=3,
        prediction=100.0,
    )

    simulator = Simulator(
        track=track,
        vehicle=vehicle,
        model=model,
        scaler=scaler,
    )

    simulator.step()

    assert (
        simulator.records[0].steering_degrees
        == pytest.approx(40.0)
    )
@pytest.mark.parametrize(
    "terminal_status",
    [
        "goal",
        "collision",
    ],
)
def test_run_stops_at_terminal_status(
    terminal_status,
):
    simulator = Simulator(
        track=None,
        vehicle=None,
        model=None,
        scaler=None,
    )

    statuses = iter([
        "running",
        "running",
        terminal_status,
    ])
    call_count = 0

    def fake_step():
        nonlocal call_count
        call_count += 1
        return next(statuses)

    simulator.step = fake_step

    result = simulator.run(max_steps=10)

    assert result == terminal_status
    assert call_count == 3
def test_run_stops_at_max_steps():
    simulator = Simulator(
        track=None,
        vehicle=None,
        model=None,
        scaler=None,
    )

    call_count = 0

    def fake_step():
        nonlocal call_count
        call_count += 1
        return "running"

    simulator.step = fake_step

    result = simulator.run(max_steps=4)

    assert result == "max_steps"
    assert call_count == 4
@pytest.mark.parametrize(
    "max_steps",
    [0, -1],
)
def test_run_rejects_invalid_max_steps(
    max_steps,
):
    simulator = Simulator(
        track=None,
        vehicle=None,
        model=None,
        scaler=None,
    )

    with pytest.raises(ValueError):
        simulator.run(max_steps=max_steps)
