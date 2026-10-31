from autocar.simulation.simulator import SimulationRecord
from pathlib import Path
import numpy as np

# simulation record: x,y, front_distance, right_distance, left_distance, steering_degrees
# 根據不同 dimention 的 model 儲存對應的 data
# 4D 前方距離、右方距離、左方距離、方向盤得出角度
# 6D x, y, 前方距離、右方距離、左方距離、方向盤得出角度

def save_4d_trajectory(path,records):
    if len(records) == 0:
        raise ValueError("records are empty")
    data = np.empty((len(records),4))
    for index, record in enumerate(records):
        data[index] = np.array([record.front_distance,record.right_distance,record.left_distance,record.steering_degrees]) 
    np.savetxt(path,data,fmt="%.7f",delimiter=" ")

def save_6d_trajectory(path,records):
    if len(records) == 0:
        raise ValueError("records are empty")
    data = np.empty((len(records),6))
    for index, record in enumerate(records):
        data[index] = np.array([record.x, record.y, record.front_distance,record.right_distance,record.left_distance,record.steering_degrees])
    np.savetxt(path, data,fmt="%.7f", delimiter=" ")

def _load_trajectory(path,expected_columns):
    path = Path(path) 
    trajectory_data = np.loadtxt(path,ndmin=2)
    if trajectory_data.shape[1] != expected_columns or not np.isfinite(trajectory_data).all():
        raise ValueError("loading trajectory Error")
    return trajectory_data
def load_4d_trajectory(path):
    return _load_trajectory(path,expected_columns=4)
def load_6d_trajectory(path):
    return _load_trajectory(path,expected_columns=6)
