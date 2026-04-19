from typing import Dict, Sequence

import numpy as np

from src.carla.geometry.transforms import dict_transform_to_matrix


def create_bbox_points(extent: Dict[str, float]) -> np.ndarray:
    ex = extent["x"]
    ey = extent["y"]
    ez = extent["z"]

    return np.array(
        [
            [ex, ey, -ez, 1.0],
            [-ex, ey, -ez, 1.0],
            [-ex, -ey, -ez, 1.0],
            [ex, -ey, -ez, 1.0],
            [ex, ey, ez, 1.0],
            [-ex, ey, ez, 1.0],
            [-ex, -ey, ez, 1.0],
            [ex, -ey, ez, 1.0],
        ],
        dtype=np.float64,
    )


def bbox_to_world_corners(obj: Dict) -> np.ndarray:
    obj_tf = dict_transform_to_matrix(
        obj["transform"]["location"],
        obj["transform"]["rotation"],
    )
    bbox_tf = dict_transform_to_matrix(
        obj["bounding_box"]["location"],
        obj["bounding_box"]["rotation"],
    )
    bbox_local_corners = create_bbox_points(obj["bounding_box"]["extent"])

    bbox_world_matrix = obj_tf @ bbox_tf
    world_corners = (bbox_world_matrix @ bbox_local_corners.T).T
    return world_corners[:, :3]


def gt_box7_to_corners_sensor(box7: Sequence[float]) -> np.ndarray:
    x, y, z, dx, dy, dz, yaw = map(float, box7)

    ex = dx / 2.0
    ey = dy / 2.0
    ez = dz / 2.0

    corners = np.array(
        [
            [ex, ey, -ez],
            [-ex, ey, -ez],
            [-ex, -ey, -ez],
            [ex, -ey, -ez],
            [ex, ey, ez],
            [-ex, ey, ez],
            [-ex, -ey, ez],
            [ex, -ey, ez],
        ],
        dtype=np.float64,
    )

    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    rotated = corners @ rot.T
    rotated += np.array([x, y, z], dtype=np.float64).reshape(1, 3)
    return rotated
