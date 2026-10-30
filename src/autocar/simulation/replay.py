from dataclasses import dataclass
from autocar.simulation.track import Track
from autocar.simulation.vehicle import Vehicle
import numpy as np

@dataclass
class ReplayFrame:
    x: float
    y: float
    heading_degrees: float
    front_distance: float
    right_distance: float
    left_distance: float
    steering_degrees: float
def replay_4d_trajectory(track, trajectory):
    # track: start_position, goal_top, goal_bottom, boundary
    # trajectory 4D 前方距離、右方距離、左方距離、方向盤得出角度
    # vehicle(x,y,heading_degrees)
    car = Vehicle(track.start_position[0], track.start_position[1], track.start_position[2])
    replay_video = []
    for front,right, left,steering in trajectory:
        frame = ReplayFrame(car.x,car.y,car.heading_degrees, front,right,left,steering)
        car.step(steering)
        replay_video.append(frame)
    return replay_video
def replay_6d_trajectory(track, trajectory):
    # track: start_position, goal_top, goal_bottom, boundary
    # trajectory 4D x, y 前方距離、右方距離、左方距離、方向盤得出角度
    # vehicle(x,y,heading_degrees)
    car = Vehicle(track.start_position[0], track.start_position[1], track.start_position[2])
    replay_video = []
    # element of [x,y, steering] 
    # [x_{t+1}, y_{t+1}] = car.step(steering)
    for x, y, front, right, left,steering in trajectory:
        frame = ReplayFrame(car.x,car.y,car.heading_degrees, front,right,left,steering)
        if not np.isclose(x,car.x) or not np.isclose(y, car.y):
            raise ValueError("trajectory's data is not the same with calculation")
        car.step(steering)
        replay_video.append(frame)
    return replay_video



    
