from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Track:
    start_position:np.ndarray
    goal_top_left:np.ndarray
    goal_bottom_right:np.ndarray
    boundary:np.ndarray

def load_track(path):
    path = Path(path).expanduser().resolve()
    with open(path, mode="r", encoding="utf-8") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    track_data = []
    for line in lines:
        data = line.replace(","," ").split()
        data = [float(value) for value in data]
        data = np.array(data,dtype=float)
        track_data.append(data)
    if track_data[0].size != 3 or track_data[1].size != 2 or track_data[2].size != 2:
        raise ValueError("start position or final boundary has been something wrong.")
    if not np.array_equal(track_data[3],track_data[-1]):
        raise ValueError("invalid track data: first position is not the same as the final position")
    track = Track(
                start_position=track_data[0],
                goal_top_left=track_data[1],
                goal_bottom_right=track_data[2],
                boundary=np.array(track_data[3:],dtype=float)
            )
    return track

if __name__ == "__main__":
    load_track("~/ML_HW_2/data/track.txt")

