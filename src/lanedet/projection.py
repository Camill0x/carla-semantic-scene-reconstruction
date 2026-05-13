from typing import Mapping

import numpy as np

from src.common.typing_aliases import Float32Array, Float64Array
from src.lanedet.prediction import Lanes2DPrediction, Lanes3DPrediction


def transform_dict_to_matrix(transform: Mapping[str, object]) -> Float64Array:
    location = transform["location"]
    rotation = transform["rotation"]
    if not isinstance(location, Mapping) or not isinstance(rotation, Mapping):
        raise ValueError("Transform must contain location and rotation mappings")

    pitch = np.radians(float(rotation["pitch"]))
    yaw = np.radians(float(rotation["yaw"]))
    roll = np.radians(float(rotation["roll"]))

    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = float(location["x"])
    matrix[1, 3] = float(location["y"])
    matrix[2, 3] = float(location["z"])

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


def build_intrinsics(width: int, height: int, fov: float) -> Float64Array:
    focal = width / (2.0 * np.tan(float(fov) * np.pi / 360.0))
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return matrix


def transform_points(points_xyz: Float64Array, transform_matrix: Float64Array) -> Float64Array:
    points_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    out = (transform_matrix @ points_h.T).T
    return np.asarray(out[:, :3], dtype=np.float64)


def world_to_lidar_matrix(lidar_transform: Mapping[str, object]) -> Float64Array:
    world_to_lidar = np.linalg.inv(transform_dict_to_matrix(lidar_transform))
    flip_y = np.eye(4, dtype=np.float64)
    flip_y[1, 1] = -1.0
    return np.asarray(flip_y @ world_to_lidar, dtype=np.float64)


def image_points_to_lidar_ground(
    points_2d: Float32Array,
    *,
    intrinsics: Float64Array,
    camera_to_world: Float64Array,
    world_to_lidar: Float64Array,
    ground_z_lidar: float,
) -> Float32Array:
    if points_2d.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    points = np.asarray(points_2d, dtype=np.float64)
    x = (points[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (points[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1]

    # Conventional camera coordinates are right/down/forward.
    # CARLA sensor coordinates are forward/right/up.
    directions_camera = np.stack(
        [np.ones_like(x), x, -y],
        axis=1,
    )
    norms = np.linalg.norm(directions_camera, axis=1, keepdims=True)
    directions_camera /= np.maximum(norms, 1e-9)

    origin_world = camera_to_world[:3, 3]
    directions_world = directions_camera @ camera_to_world[:3, :3].T
    directions_world /= np.maximum(np.linalg.norm(directions_world, axis=1, keepdims=True), 1e-9)

    origin_world_h = np.asarray([origin_world[0], origin_world[1], origin_world[2], 1.0], dtype=np.float64)
    origin_lidar = (world_to_lidar @ origin_world_h)[:3]

    points_on_ray_world = origin_world + directions_world
    points_on_ray_lidar = transform_points(points_on_ray_world, world_to_lidar)
    directions_lidar = points_on_ray_lidar - origin_lidar

    dz = directions_lidar[:, 2]
    valid = np.abs(dz) >= 1e-6
    t = np.full(points.shape[0], np.nan, dtype=np.float64)
    t[valid] = (float(ground_z_lidar) - float(origin_lidar[2])) / dz[valid]
    valid &= t > 0.0

    points_lidar = origin_lidar + t[:, None] * directions_lidar
    points_lidar[~valid] = np.nan
    return points_lidar.astype(np.float32)


def lanes_2d_to_lanes_3d(
    lanes_2d: Lanes2DPrediction,
    *,
    camera_frame: Mapping[str, object],
    state_frame: Mapping[str, object],
    score_thresh: float,
) -> Lanes3DPrediction:
    camera = camera_frame.get("camera_front") or {}
    state_camera = state_frame.get("camera_front") or {}
    lidar = state_frame.get("lidar") or {}
    if not isinstance(camera, Mapping) or not isinstance(state_camera, Mapping) or not isinstance(lidar, Mapping):
        return Lanes3DPrediction.empty()

    camera_transform = state_camera.get("transform")
    lidar_transform = lidar.get("transform")
    if camera_transform is None or lidar_transform is None:
        return Lanes3DPrediction.empty()

    image_width = int(camera.get("width", 0))
    image_height = int(camera.get("height", 0))
    camera_fov = float(state_camera.get("fov", 0.0))
    ground_z = float(lidar.get("ground_z", -1.8))
    if image_width <= 0 or image_height <= 0 or camera_fov <= 0.0:
        return Lanes3DPrediction.empty()

    strips = []
    scores = []
    names = []
    intrinsics = build_intrinsics(image_width, image_height, camera_fov)
    camera_to_world = transform_dict_to_matrix(camera_transform)
    world_to_lidar = world_to_lidar_matrix(lidar_transform)

    for index, (lane_points_2d, score) in enumerate(zip(lanes_2d.strips, lanes_2d.scores)):
        if score < score_thresh:
            continue

        lane_points_3d = image_points_to_lidar_ground(
            lane_points_2d,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            world_to_lidar=world_to_lidar,
            ground_z_lidar=ground_z,
        )
        lane_points_3d = lane_points_3d[np.isfinite(lane_points_3d).all(axis=1)]

        if len(lane_points_3d) < 2:
            continue
        strips.append(np.asarray(lane_points_3d, dtype=np.float32))
        scores.append(float(score))
        names.append(lanes_2d.names[index])

    return Lanes3DPrediction(
        strips=strips,
        scores=np.asarray(scores, dtype=np.float32),
        names=names,
    )
