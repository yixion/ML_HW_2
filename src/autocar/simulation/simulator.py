from autocar.ml.mlp import MLPRegressor
from autocar.ml.preprocessing import StandardScaler
from autocar.simulation.sensors import measure_sensor_distances
from autocar.simulation.vehicle import Vehicle
from autocar.simulation.collision import (vehicle_collides,vehicle_reached_goal)
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationRecord:
    x:float
    y:float
    front_distance:float
    right_distance:float
    left_distance:float
    steering_degrees:float

class Simulator:
    def __init__(
        self,
        track,
        vehicle,
        model,
        scaler,
    ):
        self.track = track
        self.vehicle = vehicle
        self.model = model
        self.scaler = scaler
        self.records = []

    def step(self):
        print("current position of vehicle:",self.vehicle.x,self.vehicle.y)
        print("current boundary:",self.track.boundary)
        # 建立 input
        # 取得：前面、右方、左方的 distances
        cur_distances = measure_sensor_distances(self.vehicle,self.track.boundary)
        # 根據模型輸入維度建立 feature
        input_size = self.model.layer_sizes[0]
        if input_size == 3:
            features = np.array([cur_distances[0],
                                 cur_distances[1],
                                 cur_distances[2]
                                 ])
        elif input_size == 5:
            features = np.array([self.vehicle.x,
                                 self.vehicle.y,
                                 cur_distances[0],
                                 cur_distances[1],
                                 cur_distances[2]
                                 ])
        # 調整模型的維度
        features = features.reshape(1,features.shape[0])
        # 正規化 inputs
        features = self.scaler.transform(features)
        # 用模型預測 steering_degrees
        prediction = self.model.predict(features)
        prediction = np.clip(prediction,-40.0,40.0)
        prediction = prediction.item()
        # 紀錄狀態
        record = SimulationRecord(
                    x=self.vehicle.x,
                    y=self.vehicle.y,
                    front_distance=cur_distances[0],
                    right_distance=cur_distances[1],
                    left_distance=cur_distances[2],
                    steering_degrees=prediction
                    )
        self.records.append(record)
        # 車子前進
        self.vehicle.step(prediction)
        if vehicle_collides(self.vehicle,self.track.boundary):
            return "collision"
        elif vehicle_reached_goal(self.vehicle, self.track.goal_top_left,self.track.goal_bottom_right):
            return "goal"
        else:
            return "running"

    def run(self,max_steps=1000):
        if max_steps<=0:
            raise ValueError("pls enter psotive steps")
        print("Start simulation...")
        for i in range(max_steps):
            print(f"step{i}")
            status = self.step()
            if status == "goal":
                return "goal"
            elif status == "collision":
                return "collision"
        return "max_steps"


