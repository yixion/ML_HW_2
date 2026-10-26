import numpy as np
import pytest

from autocar.simulation.collision import (
    vehicle_collides,
    vehicle_reached_goal,
)
from autocar.simulation.vehicle import Vehicle


@pytest.mark.parametrize(
    "x, expected",
    [
        # 距離右牆 10，大於 radius 3
        (0.0, False),

        # 距離右牆 2，小於 radius 3
        (8.0, True),

        # 距離右牆剛好為 3，只有相切，允許
        (7.0, False),
    ],
)
def test_vehicle_collision_with_boundary(
    x,
    expected,
):
    boundary = np.array([
        [-10.0, -10.0],
        [10.0, -10.0],
        [10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, -10.0],
    ])

    vehicle = Vehicle(
        x=x,
        y=0.0,
        heading_degrees=90.0,
        radius=3.0,
    )

    result = vehicle_collides(
        vehicle,
        boundary,
    )

    assert result is expected


@pytest.mark.parametrize(
    "x, y, expected",
    [
        # 終點矩形內
        (0.0, 4.0, True),

        # 位於終點矩形邊界
        (2.0, 3.0, True),

        # 超出右側
        (3.0, 4.0, False),

        # 超出上方
        (0.0, 6.0, False),
    ],
)
def test_vehicle_reached_goal(
    x,
    y,
    expected,
):
    vehicle = Vehicle(
        x=x,
        y=y,
        heading_degrees=90.0,
    )

    goal_top_left = np.array([
        -2.0,
        5.0,
    ])
    goal_bottom_right = np.array([
        2.0,
        3.0,
    ])

    result = vehicle_reached_goal(
        vehicle,
        goal_top_left,
        goal_bottom_right,
    )

    assert result is expected
