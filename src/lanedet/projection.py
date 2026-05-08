from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def transform_dict_to_matrix(transform: Mapping[str, object]) -> np.ndarray:
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


def build_intrinsics(width: int, height: int, fov: float) -> np.ndarray:
    focal = width / (2.0 * np.tan(float(fov) * np.pi / 360.0))
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return matrix


def transform_points(points_xyz: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    points_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    out = (transform_matrix @ points_h.T).T
    return out[:, :3]


def world_points_to_lidar(points_world: np.ndarray, lidar_transform: Mapping[str, object]) -> np.ndarray:
    lidar_to_world = transform_dict_to_matrix(lidar_transform)
    world_to_lidar = np.linalg.inv(lidar_to_world)
    points_lidar = transform_points(points_world, world_to_lidar)
    points_lidar[:, 1] *= -1.0
    return points_lidar.astype(np.float32)


def image_point_to_lidar_ground(
    point_2d: Sequence[float],
    *,
    intrinsics: np.ndarray,
    camera_transform: Mapping[str, object],
    lidar_transform: Mapping[str, object],
    ground_z_lidar: float,
) -> np.ndarray:
    u, v = float(point_2d[0]), float(point_2d[1])
    x = (u - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) / intrinsics[1, 1]

    # Conventional camera coordinates are right/down/forward. CARLA sensor
    # coordinates are forward/right/up.
    direction_camera = np.asarray([1.0, x, -y], dtype=np.float64)
    direction_camera /= max(float(np.linalg.norm(direction_camera)), 1e-9)

    camera_to_world = transform_dict_to_matrix(camera_transform)
    origin_world = camera_to_world[:3, 3]
    direction_world = camera_to_world[:3, :3] @ direction_camera
    direction_world /= max(float(np.linalg.norm(direction_world)), 1e-9)

    origin_lidar = world_points_to_lidar(origin_world.reshape(1, 3), lidar_transform)[0].astype(np.float64)
    point_on_ray_world = origin_world + direction_world
    point_on_ray_lidar = world_points_to_lidar(point_on_ray_world.reshape(1, 3), lidar_transform)[0].astype(np.float64)
    direction_lidar = point_on_ray_lidar - origin_lidar

    if abs(float(direction_lidar[2])) < 1e-6:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)

    t = (float(ground_z_lidar) - float(origin_lidar[2])) / float(direction_lidar[2])
    if t <= 0.0:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)

    point_lidar = origin_lidar + t * direction_lidar
    return point_lidar.astype(np.float32)


def lanes_2d_to_lanes_3d_payload(
    lanes_2d: List[Tuple[np.ndarray, float]],
    *,
    camera_frame: Mapping[str, object],
    state_frame: Mapping[str, object],
    score_thresh: float,
) -> Dict[str, object]:
    camera = camera_frame.get("camera_front") or {}
    state_camera = state_frame.get("camera_front") or {}
    lidar = state_frame.get("lidar") or {}
    if not isinstance(camera, Mapping) or not isinstance(state_camera, Mapping) or not isinstance(lidar, Mapping):
        return {"strips": [], "scores": [], "names": [], "projection": "missing_sensor_state"}

    camera_transform = state_camera.get("transform")
    lidar_transform = lidar.get("transform")
    if camera_transform is None or lidar_transform is None:
        return {"strips": [], "scores": [], "names": [], "projection": "missing_transforms"}

    image_width = int(camera.get("width", 0))
    image_height = int(camera.get("height", 0))
    camera_fov = float(state_camera.get("fov", 0.0))
    ground_z = float(lidar.get("ground_z", -1.8))
    if image_width <= 0 or image_height <= 0 or camera_fov <= 0.0:
        return {"strips": [], "scores": [], "names": [], "projection": "missing_camera_intrinsics"}

    strips = []
    scores = []
    names = []
    intrinsics = build_intrinsics(image_width, image_height, camera_fov)

    for index, (lane_points_2d, score) in enumerate(lanes_2d):
        if score < score_thresh:
            continue

        lane_points_3d = []
        for point_2d in lane_points_2d:
            point_lidar = image_point_to_lidar_ground(
                point_2d,
                intrinsics=intrinsics,
                camera_transform=camera_transform,
                lidar_transform=lidar_transform,
                ground_z_lidar=ground_z,
            )
            if np.isfinite(point_lidar).all():
                lane_points_3d.append(point_lidar)

        if len(lane_points_3d) < 2:
            continue
        strips.append(np.asarray(lane_points_3d, dtype=np.float32))
        scores.append(float(score))
        names.append(f"lane_{index}")

    return {
        "strips": strips,
        "scores": scores,
        "names": names,
        "projection": "flat_ground_lidar",
    }
