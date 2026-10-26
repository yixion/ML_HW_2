import numpy as np

from autocar.simulation.sensors import (
    measure_sensor_distances,
)
from autocar.simulation.vehicle import Vehicle


def test_measure_sensor_distances_in_rectangle():
    boundary = np.array([
        [-5.0, -5.0],
        [5.0, -5.0],
        [5.0, 10.0],
        [-5.0, 10.0],
        [-5.0, -5.0],
    ])

    vehicle = Vehicle(
        x=0.0,
        y=0.0,
        heading_degrees=90.0,
    )

    distances = measure_sensor_distances(
        vehicle,
        boundary,
    )

    expected = np.array([
        10.0,          # front：碰到 y=10
        np.sqrt(50.0), # right：碰到 x=5
        np.sqrt(50.0), # left：碰到 x=-5
    ])

    assert distances.shape == (3,)
    np.testing.assert_allclose(
        distances,
        expected,
    )
