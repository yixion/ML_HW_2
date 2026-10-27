import pytest
import numpy as np

from autocar.simulation.simulator import (
    SimulationRecord,
)
from autocar.storage.trajectory import (
    load_4d_trajectory,
    load_6d_trajectory,
    save_4d_trajectory,
    save_6d_trajectory,
)


@pytest.fixture
def records():
    return [
        SimulationRecord(
            x=1.0,
            y=2.0,
            front_distance=10.0,
            right_distance=7.5,
            left_distance=8.25,
            steering_degrees=2.0,
        ),
        SimulationRecord(
            x=1.5,
            y=3.0,
            front_distance=9.0,
            right_distance=7.0,
            left_distance=8.0,
            steering_degrees=-1.5,
        ),
    ]
def test_save_4d_trajectory(
    tmp_path,
    records,
):
    output_path = tmp_path / "track4D.txt"

    save_4d_trajectory(
        output_path,
        records,
    )

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert lines == [
        (
            "10.0000000 7.5000000 "
            "8.2500000 2.0000000"
        ),
        (
            "9.0000000 7.0000000 "
            "8.0000000 -1.5000000"
        ),
    ]
def test_save_6d_trajectory(
    tmp_path,
    records,
):
    output_path = tmp_path / "track6D.txt"

    save_6d_trajectory(
        output_path,
        records,
    )

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert lines == [
        (
            "1.0000000 2.0000000 "
            "10.0000000 7.5000000 "
            "8.2500000 2.0000000"
        ),
        (
            "1.5000000 3.0000000 "
            "9.0000000 7.0000000 "
            "8.0000000 -1.5000000"
        ),
    ]
@pytest.mark.parametrize(
    "save_function, filename",
    [
        (
            save_4d_trajectory,
            "track4D.txt",
        ),
        (
            save_6d_trajectory,
            "track6D.txt",
        ),
    ],
)
def test_save_trajectory_rejects_empty_records(
    tmp_path,
    save_function,
    filename,
):
    with pytest.raises(ValueError):
        save_function(
            tmp_path / filename,
            [],
        )
@pytest.mark.parametrize(
    "load_function, column_count",
    [
        (load_4d_trajectory, 4),
        (load_6d_trajectory, 6),
    ],
)
def test_load_trajectory_returns_2d_array(
    tmp_path,
    load_function,
    column_count,
):
    path = tmp_path / "trajectory.txt"
    expected = np.arange(
        column_count,
        dtype=float,
    ).reshape(1, column_count)

    np.savetxt(path, expected)

    actual = load_function(path)

    assert actual.shape == (1, column_count)
    np.testing.assert_allclose(
        actual,
        expected,
    )
@pytest.mark.parametrize(
    "load_function, wrong_column_count",
    [
        (load_4d_trajectory, 3),
        (load_6d_trajectory, 5),
    ],
)
def test_load_trajectory_rejects_wrong_columns(
    tmp_path,
    load_function,
    wrong_column_count,
):
    path = tmp_path / "invalid_trajectory.txt"
    invalid_data = np.zeros(
        (2, wrong_column_count),
    )

    np.savetxt(path, invalid_data)

    with pytest.raises(ValueError):
        load_function(path)
