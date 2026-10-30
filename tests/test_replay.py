import numpy as np
import pytest

from autocar.simulation.replay import (
    replay_4d_trajectory,
    replay_6d_trajectory,
)
from autocar.simulation.track import Track


@pytest.fixture
def track():
    return Track(
        start_position=np.array([
            0.0,
            0.0,
            90.0,
        ]),
        goal_top_left=np.array([
            -1.0,
            10.0,
        ]),
        goal_bottom_right=np.array([
            1.0,
            8.0,
        ]),
        boundary=np.array([
            [-5.0, -3.0],
            [5.0, -3.0],
            [5.0, 12.0],
            [-5.0, 12.0],
            [-5.0, -3.0],
        ]),
    )
def test_replay_4d_reconstructs_vehicle_positions(
    track,
):
    trajectory = np.array([
        [10.0, 5.0, 6.0, 0.0],
        [9.0, 4.0, 5.0, 0.0],
    ])

    frames = replay_4d_trajectory(
        track,
        trajectory,
    )

    assert len(frames) == 2

    first = frames[0]
    assert first.x == pytest.approx(0.0)
    assert first.y == pytest.approx(0.0)
    assert first.heading_degrees == pytest.approx(
        90.0
    )
    assert first.front_distance == pytest.approx(
        10.0
    )
    assert first.right_distance == pytest.approx(
        5.0
    )
    assert first.left_distance == pytest.approx(
        6.0
    )
    assert first.steering_degrees == pytest.approx(
        0.0
    )

    second = frames[1]
    assert second.x == pytest.approx(0.0)
    assert second.y == pytest.approx(1.0)
    assert second.heading_degrees == pytest.approx(
        90.0
    )
def test_replay_6d_uses_recorded_positions(
    track,
):
    trajectory = np.array([
        [
            0.0,
            0.0,
            10.0,
            5.0,
            6.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            9.0,
            4.0,
            5.0,
            0.0,
        ],
    ])

    frames = replay_6d_trajectory(
        track,
        trajectory,
    )

    assert len(frames) == 2

    assert frames[0].x == pytest.approx(0.0)
    assert frames[0].y == pytest.approx(0.0)

    assert frames[1].x == pytest.approx(0.0)
    assert frames[1].y == pytest.approx(1.0)

    assert (
        frames[1].front_distance
        == pytest.approx(9.0)
    )
    assert (
        frames[1].right_distance
        == pytest.approx(4.0)
    )
    assert (
        frames[1].left_distance
        == pytest.approx(5.0)
    )
