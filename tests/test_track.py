import numpy as np
import pytest

from autocar.simulation.track import load_track


def test_load_track_returns_expected_data(
    tmp_path,
):
    track_path = tmp_path / "track.txt"
    track_path.write_text(
        "\n".join([
            "0, 0, 90",
            "-2, 10",
            "2, 8",
            "-6, -3",
            "6, -3",
            "6, 12",
            "-6, 12",
            "-6, -3",
        ]),
        encoding="utf-8",
    )

    track = load_track(track_path)

    np.testing.assert_allclose(
        track.start_position,
        [0.0, 0.0, 90.0],
    )
    np.testing.assert_allclose(
        track.goal_top_left,
        [-2.0, 10.0],
    )
    np.testing.assert_allclose(
        track.goal_bottom_right,
        [2.0, 8.0],
    )

    assert track.boundary.shape == (5, 2)
    np.testing.assert_allclose(
        track.boundary[0],
        track.boundary[-1],
    )
def test_load_track_rejects_open_boundary(
    tmp_path,
):
    track_path = tmp_path / "open_track.txt"
    track_path.write_text(
        "\n".join([
            "0 0 90",
            "-2 10",
            "2 8",
            "-6 -3",
            "6 -3",
            "6 12",
            "-6 12",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_track(track_path)

