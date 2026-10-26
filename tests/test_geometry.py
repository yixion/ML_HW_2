import numpy as np
import pytest

from autocar.simulation.geometry import (
    ray_segment_intersection_distance,
    point_to_segment_distance,
)


def test_ray_intersects_segment():
    distance = ray_segment_intersection_distance(
        ray_origin=np.array([0.0, 0.0]),
        ray_direction=np.array([1.0, 0.0]),
        segment_start=np.array([5.0, -2.0]),
        segment_end=np.array([5.0, 2.0]),
    )

    assert distance == pytest.approx(5.0)

@pytest.mark.parametrize(
    "segment_start, segment_end",
    [
        # 牆壁位於射線後方
        (
            np.array([-5.0, -2.0]),
            np.array([-5.0, 2.0]),
        ),

        # 牆壁與射線平行
        (
            np.array([0.0, 1.0]),
            np.array([5.0, 1.0]),
        ),

        # 無限延伸後會相交，但交點不在線段內
        (
            np.array([5.0, 2.0]),
            np.array([5.0, 4.0]),
        ),
    ],
)
def test_ray_returns_none_without_valid_intersection(
    segment_start,
    segment_end,
):
    distance = ray_segment_intersection_distance(
        ray_origin=np.array([0.0, 0.0]),
        ray_direction=np.array([1.0, 0.0]),
        segment_start=segment_start,
        segment_end=segment_end,
    )

    assert distance is None
@pytest.mark.parametrize(
    "point, expected_distance",
    [
        # 投影在線段中間，最近點為 (4, 0)
        (
            np.array([4.0, 3.0]),
            3.0,
        ),

        # 投影在線段起點左側，最近點為 (0, 0)
        (
            np.array([-2.0, 3.0]),
            np.sqrt(13.0),
        ),

        # 投影在線段終點右側，最近點為 (10, 0)
        (
            np.array([12.0, 3.0]),
            np.sqrt(13.0),
        ),
    ],
)
def test_point_to_segment_distance(
    point,
    expected_distance,
):
    distance = point_to_segment_distance(
        point=point,
        segment_start=np.array([0.0, 0.0]),
        segment_end=np.array([10.0, 0.0]),
    )

    assert distance == pytest.approx(
        expected_distance
    )
