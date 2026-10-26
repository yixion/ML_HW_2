import numpy as np

def cross_2d(first, second):
    return(first[0] * second[1] - first[1] * second[0])
def ray_segment_intersection_distance(
        ray_origin,
        ray_direction,
        segment_start,
        segment_end,
        ):
    s = segment_end - segment_start
    difference = segment_start - ray_origin
    denominator = cross_2d(ray_direction,s)
    if np.isclose(denominator,0.0):
        print("no intersection")
        return None
    t = cross_2d(difference,s)/denominator
    u = cross_2d(difference,ray_direction)/denominator
    if t >= 0.0 and u >= 0.0 and u <= 1.0:
        return t
    else:
        #print(f"{t} and {u} intersection is not on this line segment")
        None


