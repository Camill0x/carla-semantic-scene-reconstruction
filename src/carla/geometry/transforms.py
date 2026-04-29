from typing import Sequence

import numpy as np

import carla


def transform_points(points_xyz: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    points_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    out = (transform_matrix @ points_h.T).T
    return out[:, :3]


def distance_between_locations(loc1: carla.Location, loc2: carla.Location) -> float:
    return float(loc1.distance(loc2))


def carla_transform_to_matrix(transform: carla.Transform) -> np.ndarray:
    return np.array(transform.get_matrix(), dtype=np.float64)


def world_to_sensor(points_world_xyz: np.ndarray, sensor_transform: carla.Transform) -> np.ndarray:
    world_to_sensor_tf = np.linalg.inv(carla_transform_to_matrix(sensor_transform))
    return transform_points(points_world_xyz, world_to_sensor_tf)


def world_to_sensor_point(point_world_xyz: Sequence[float], sensor_transform: carla.Transform) -> np.ndarray:
    pts = np.asarray(point_world_xyz, dtype=np.float64).reshape(1, 3)
    return world_to_sensor(pts, sensor_transform)[0]
