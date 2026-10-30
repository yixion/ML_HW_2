from dataclasses import dataclass
import numpy as np

@dataclass
class Vehicle:
    x: float
    y: float
    heading_degrees: float
    length: float = 6.0
    radius: float = 3.0
    
    def step(self,steering_degrees):
        if steering_degrees > 40.0 or steering_degrees < -40.0:
            raise ValueError("方向盤壞了")
        heading_radians = np.deg2rad(self.heading_degrees)
        steering_radians = np.deg2rad(steering_degrees)
        self.x = self.x + np.cos(heading_radians+steering_radians) + np.sin(steering_radians)*np.sin(heading_radians) 
        self.x = self.x.item()
        self.y = self.y + np.sin(heading_radians+steering_radians) - np.sin(steering_radians)*np.cos(heading_radians) 
        self.y = self.y.item()
        heading_radians = heading_radians - np.arcsin((2*np.sin(steering_radians))/self.length)
        self.heading_degrees = np.rad2deg(heading_radians)

