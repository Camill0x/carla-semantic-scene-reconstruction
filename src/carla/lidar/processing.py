import numpy as np

import carla
from src.carla.geometry.boxes import filter_points_inside_ego_vehicle


def flip_lidar_y(points: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[:, 1] *= -1.0
    return out


def preprocess_lidar_points(
    points: np.ndarray,
    hero: carla.Actor,
    lidar: carla.Sensor,
    ego_bbox_padding: float,
) -> np.ndarray:
    points = flip_lidar_y(points)
    points = filter_points_inside_ego_vehicle(
        points=points,
        hero=hero,
        lidar=lidar,
        padding=ego_bbox_padding,
    )
    return points
