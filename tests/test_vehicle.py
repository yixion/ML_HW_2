import numpy as np
import pytest

from autocar.simulation.vehicle import Vehicle


def test_vehicle_moves_forward_with_zero_steering():
    vehicle = Vehicle(
        x=0.0,
        y=0.0,
        heading_degrees=90.0,
    )

    vehicle.step(steering_degrees=0.0)

    assert vehicle.x == pytest.approx(0.0)
    assert vehicle.y == pytest.approx(1.0)
    assert vehicle.heading_degrees == pytest.approx(90.0)
def test_vehicle_turns_with_positive_steering():
    vehicle = Vehicle(
        x=0.0,
        y=0.0,
        heading_degrees=90.0,
        length=6.0,
    )

    vehicle.step(steering_degrees=30.0)

    expected_heading = 90.0 - np.rad2deg(
        np.arcsin(1.0 / 6.0)
    )

    assert vehicle.x == pytest.approx(0.0)
    assert vehicle.y == pytest.approx(
        np.sqrt(3.0) / 2.0
    )
    assert vehicle.heading_degrees == pytest.approx(
        expected_heading
    )
@pytest.mark.parametrize(
    "steering_degrees",
    [-40.1, 40.1],
)
def test_vehicle_rejects_invalid_steering(
    steering_degrees,
):
    vehicle = Vehicle(
        x=0.0,
        y=0.0,
        heading_degrees=90.0,
    )

    with pytest.raises(ValueError):
        vehicle.step(steering_degrees)

