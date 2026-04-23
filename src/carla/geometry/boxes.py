import math
from typing import Dict

import numpy as np

import carla
from src.carla.geometry.transforms import (
    carla_transform_to_matrix,
    world_to_sensor_point,
)


def points_inside_oriented_box(
    points_xyz: np.ndarray,
    center_xyz: np.ndarray,
    yaw_rad: float,
    half_sizes_xyz: np.ndarray,
) -> np.ndarray:
    rel = points_xyz - center_xyz.reshape(1, 3)

    c = math.cos(-yaw_rad)
    s = math.sin(-yaw_rad)
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    local = rel @ rot.T

    return (
        (np.abs(local[:, 0]) <= half_sizes_xyz[0])
        & (np.abs(local[:, 1]) <= half_sizes_xyz[1])
        & (np.abs(local[:, 2]) <= half_sizes_xyz[2])
    )


def get_ego_box_in_lidar_frame(
    hero: carla.Actor,
    lidar: carla.Sensor,
    padding: float = 0.15,
) -> Dict[str, np.ndarray]:
    bbox = hero.bounding_box
    actor_tf = carla_transform_to_matrix(hero.get_transform())
    bbox_center_local = np.array(
        [[bbox.location.x], [bbox.location.y], [bbox.location.z], [1.0]],
        dtype=np.float64,
    )
    bbox_center_world = (actor_tf @ bbox_center_local)[:3, 0]

    bbox_yaw_world = hero.get_transform().rotation.yaw + bbox.rotation.yaw
    lidar_yaw_world = lidar.get_transform().rotation.yaw
    yaw_sensor = math.radians(bbox_yaw_world - lidar_yaw_world)

    center_sensor = world_to_sensor_point(bbox_center_world, lidar.get_transform())
    center_sensor[1] *= -1.0

    half_sizes = np.array(
        [
            float(bbox.extent.x + padding),
            float(bbox.extent.y + padding),
            float(bbox.extent.z + padding),
        ],
        dtype=np.float64,
    )

    yaw_sensor = -yaw_sensor

    return {
        "center": center_sensor,
        "half_sizes": half_sizes,
        "yaw": np.array(yaw_sensor, dtype=np.float64),
    }


def filter_points_inside_ego_vehicle(
    points: np.ndarray,
    hero: carla.Actor,
    lidar: carla.Sensor,
) -> np.ndarray:
    ego_box = get_ego_box_in_lidar_frame(hero, lidar)
    inside = points_inside_oriented_box(
        points_xyz=points[:, :3],
        center_xyz=ego_box["center"],
        yaw_rad=float(ego_box["yaw"]),
        half_sizes_xyz=ego_box["half_sizes"],
    )
    return points[~inside]


def get_bbox_center_world_from_actor(actor: carla.Actor) -> np.ndarray:
    bbox = actor.bounding_box
    actor_tf = carla_transform_to_matrix(actor.get_transform())
    center_local = np.array(
        [[bbox.location.x], [bbox.location.y], [bbox.location.z], [1.0]],
        dtype=np.float64,
    )
    return (actor_tf @ center_local)[:3, 0]


def actor_matches_level_bbox(
    actor: carla.Actor,
    level_bbox: carla.BoundingBox,
    center_thresh: float = 1.5,
    extent_thresh: float = 1.0,
) -> bool:
    actor_center = get_bbox_center_world_from_actor(actor)
    level_center = np.array(
        [level_bbox.location.x, level_bbox.location.y, level_bbox.location.z],
        dtype=np.float64,
    )
    center_distance = np.linalg.norm(actor_center - level_center)

    actor_extent = np.array(
        [actor.bounding_box.extent.x, actor.bounding_box.extent.y, actor.bounding_box.extent.z],
        dtype=np.float64,
    )
    level_extent = np.array(
        [level_bbox.extent.x, level_bbox.extent.y, level_bbox.extent.z],
        dtype=np.float64,
    )
    extent_distance = np.linalg.norm(actor_extent - level_extent)

    return center_distance <= center_thresh and extent_distance <= extent_thresh


def get_yaw_in_lidar(obj_yaw_world_deg: float, lidar_yaw_world_deg: float) -> float:
    relative_yaw_deg = obj_yaw_world_deg - lidar_yaw_world_deg
    yaw_rad = math.radians(relative_yaw_deg)
    yaw_rad = -yaw_rad
    return float(yaw_rad)


def actor_to_gt_box(actor: carla.Actor, lidar_transform: carla.Transform) -> np.ndarray:
    bbox = actor.bounding_box

    bbox_local = np.array(
        [[bbox.location.x], [bbox.location.y], [bbox.location.z], [1.0]],
        dtype=np.float64,
    )
    actor_world_matrix = carla_transform_to_matrix(actor.get_transform())
    bbox_center_world = actor_world_matrix @ bbox_local

    lidar_world_matrix = carla_transform_to_matrix(lidar_transform)
    world_to_lidar = np.linalg.inv(lidar_world_matrix)
    bbox_center_lidar = world_to_lidar @ bbox_center_world

    x = float(bbox_center_lidar[0, 0])
    y = float(-bbox_center_lidar[1, 0])
    z = float(bbox_center_lidar[2, 0])

    dx = float(bbox.extent.x * 2.0)
    dy = float(bbox.extent.y * 2.0)
    dz = float(bbox.extent.z * 2.0)

    yaw = get_yaw_in_lidar(
        obj_yaw_world_deg=actor.get_transform().rotation.yaw + bbox.rotation.yaw,
        lidar_yaw_world_deg=lidar_transform.rotation.yaw,
    )

    return np.array([x, y, z, dx, dy, dz, yaw], dtype=np.float32)


def level_bbox_to_gt_box(level_bbox: carla.BoundingBox, lidar_transform: carla.Transform) -> np.ndarray:
    center_world = np.array(
        [[level_bbox.location.x], [level_bbox.location.y], [level_bbox.location.z], [1.0]],
        dtype=np.float64,
    )

    lidar_world_matrix = carla_transform_to_matrix(lidar_transform)
    world_to_lidar = np.linalg.inv(lidar_world_matrix)
    center_lidar = world_to_lidar @ center_world

    x = float(center_lidar[0, 0])
    y = float(-center_lidar[1, 0])
    z = float(center_lidar[2, 0])

    dx = float(level_bbox.extent.x * 2.0)
    dy = float(level_bbox.extent.y * 2.0)
    dz = float(level_bbox.extent.z * 2.0)

    yaw = get_yaw_in_lidar(
        obj_yaw_world_deg=level_bbox.rotation.yaw,
        lidar_yaw_world_deg=lidar_transform.rotation.yaw,
    )

    return np.array([x, y, z, dx, dy, dz, yaw], dtype=np.float32)


# def count_points_in_box7(points_xyz: np.ndarray, box7: np.ndarray) -> int:
#     x, y, z, dx, dy, dz, yaw = map(float, box7)

#     rel = points_xyz - np.array([x, y, z], dtype=np.float64).reshape(1, 3)

#     c = math.cos(-yaw)
#     s = math.sin(-yaw)
#     rot = np.array(
#         [
#             [c, -s, 0.0],
#             [s, c, 0.0],
#             [0.0, 0.0, 1.0],
#         ],
#         dtype=np.float64,
#     )
#     local = rel @ rot.T

#     inside = (np.abs(local[:, 0]) <= dx / 2.0) & (np.abs(local[:, 1]) <= dy / 2.0) & (np.abs(local[:, 2]) <= dz / 2.0)
#     return int(np.count_nonzero(inside))
