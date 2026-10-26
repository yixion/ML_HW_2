from autocar.simulation.vehicle import Vehicle
from autocar.simulation.geometry import point_to_segment_distance
import numpy as np

def vehicle_collides(vehicle,boundary):

    for segment_start, segment_end in zip(boundary[:-1],boundary[1:]):
        point = np.array([vehicle.x, vehicle.y])
        distance = point_to_segment_distance(
                point = point,
                segment_start = segment_start,
                segment_end = segment_end
            )
        if vehicle.radius > distance:
            return True
    return False

def vehicle_reached_goal(
        vehicle,
        goal_top_left,
        goal_bottom_right,
        ):
    min_x = goal_top_left[0]
    max_x = goal_bottom_right[0]
    min_y = goal_bottom_right[1]
    max_y = goal_top_left[1]
    if (vehicle.x >=min_x 
        and vehicle.x <=max_x 
        and vehicle.y >=min_y 
        and vehicle.y <=max_y
        ):
        return True
    else:
        return False
    
