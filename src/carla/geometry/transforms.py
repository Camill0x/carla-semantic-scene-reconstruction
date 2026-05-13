from typing import Sequence

import numpy as np

import carla
from src.common.typing_aliases import Float64Array


def transform_points(points_xyz: Float64Array, transform_matrix: Float64Array) -> Float64Array:
    """Transform 3D points with a homogeneous transform matrix."""
    points_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    out = (transform_matrix @ points_h.T).T
    return np.asarray(out[:, :3], dtype=np.float64)


def distance_between_locations(loc1: carla.Location, loc2: carla.Location) -> float:
    """Return the Euclidean distance between two CARLA locations."""
    return float(loc1.distance(loc2))


def carla_transform_to_matrix(transform: carla.Transform) -> Float64Array:
    """Convert a CARLA transform into a 4x4 NumPy matrix."""
    return np.asarray(transform.get_matrix(), dtype=np.float64)


def world_to_sensor(points_world_xyz: Float64Array, sensor_transform: carla.Transform) -> Float64Array:
    """Transform world-space points into a sensor coordinate frame."""
    world_to_sensor_tf = np.asarray(
        np.linalg.inv(carla_transform_to_matrix(sensor_transform)),
        dtype=np.float64,
    )
    return transform_points(points_world_xyz, world_to_sensor_tf)


def world_to_sensor_point(point_world_xyz: Sequence[float], sensor_transform: carla.Transform) -> Float64Array:
    """Transform one world-space point into a sensor coordinate frame."""
    pts = np.asarray(point_world_xyz, dtype=np.float64).reshape(1, 3)
    return np.asarray(world_to_sensor(pts, sensor_transform)[0], dtype=np.float64)
