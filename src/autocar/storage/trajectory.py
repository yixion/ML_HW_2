from autocar.simulation.simulator import SimulationRecord
import numpy as np


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
