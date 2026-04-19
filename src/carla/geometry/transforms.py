from typing import Dict, Sequence

import numpy as np

import carla

# =========================
# helpers
# =========================


def transform_points(points_xyz: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    points_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    out = (transform_matrix @ points_h.T).T
    return out[:, :3]


def distance_between_locations(loc1: carla.Location, loc2: carla.Location) -> float:
    return float(loc1.distance(loc2))


# =========================
# CARLA API -> matrix
# =========================


def carla_transform_to_matrix(transform: carla.Transform) -> np.ndarray:
    return np.array(transform.get_matrix(), dtype=np.float64)


def world_to_sensor(points_world_xyz: np.ndarray, sensor_transform: carla.Transform) -> np.ndarray:
    world_to_sensor_tf = np.linalg.inv(carla_transform_to_matrix(sensor_transform))
    return transform_points(points_world_xyz, world_to_sensor_tf)


def world_to_sensor_point(point_world_xyz: Sequence[float], sensor_transform: carla.Transform) -> np.ndarray:
    pts = np.asarray(point_world_xyz, dtype=np.float64).reshape(1, 3)
    return world_to_sensor(pts, sensor_transform)[0]


# =========================
# dict/json -> matrix
# =========================


def dict_transform_to_matrix(location: Dict[str, float], rotation: Dict[str, float]) -> np.ndarray:
    pitch = np.radians(rotation["pitch"])
    yaw = np.radians(rotation["yaw"])
    roll = np.radians(rotation["roll"])

    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = location["x"]
    matrix[1, 3] = location["y"]
    matrix[2, 3] = location["z"]

    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r

    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r

    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def world_to_sensor_dict(
    points_world_xyz: np.ndarray,
    sensor_location: Dict[str, float],
    sensor_rotation: Dict[str, float],
) -> np.ndarray:
    sensor_tf = dict_transform_to_matrix(sensor_location, sensor_rotation)
    world_to_sensor_tf = np.linalg.inv(sensor_tf)
    return transform_points(points_world_xyz, world_to_sensor_tf)


def world_to_sensor_point_dict(
    point_world_xyz: Sequence[float],
    sensor_location: Dict[str, float],
    sensor_rotation: Dict[str, float],
) -> np.ndarray:
    pts = np.asarray(point_world_xyz, dtype=np.float64).reshape(1, 3)
    return world_to_sensor_dict(pts, sensor_location, sensor_rotation)[0]
