from autocar.simulation.vehicle import Vehicle
from autocar.simulation.geometry import ray_segment_intersection_distance
import numpy as np

def measure_sensor_distance(
        vehicle,
        boundary,
        angle_offset_degrees,
        ):
    ray_origin = np.array([vehicle.x,vehicle.y])
    sensor_angle = vehicle.heading_degrees + angle_offset_degrees
    sensor_angle_radiant = np.deg2rad(sensor_angle)
    ray_direction = np.array([np.cos(sensor_angle_radiant),np.sin(sensor_angle_radiant)])
    min_distance = None
    for segment_start, segment_end in zip (boundary[:-1],boundary[1:]):
        t = ray_segment_intersection_distance(
                ray_origin = ray_origin,
                ray_direction = ray_direction,
                segment_start = segment_start,
                segment_end = segment_end
                )
        if t is not None:
            if min_distance is None:
                min_distance = t
            else:
                min_distance = min(min_distance, t)
    if min_distance is None:
        raise RuntimeError("something went wrong becuase there is no distnace data")
    return min_distance

def measure_sensor_distances(vehicle,boundary):
    return np.array([measure_sensor_distance(vehicle,boundary,0.0), measure_sensor_distance(vehicle,boundary,-45.0), measure_sensor_distance(vehicle,boundary,45.0)])
