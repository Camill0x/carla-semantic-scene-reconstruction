import numpy as np

import carla
from src.carla.geometry.boxes import filter_points_inside_ego_vehicle
from src.common.typing_aliases import Float64Array


def flip_lidar_y(points: Float64Array) -> Float64Array:
    out = np.asarray(points.copy(), dtype=np.float64)
    out[:, 1] *= -1.0
    return out


def preprocess_lidar_points(
    points: Float64Array,
    hero: carla.Actor,
    lidar: carla.Sensor,
) -> Float64Array:
    points = flip_lidar_y(points)
    points = filter_points_inside_ego_vehicle(
        points=points,
        hero=hero,
        lidar=lidar,
    )
    return points
