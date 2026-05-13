from typing import List, Optional, Tuple

import numpy as np

import carla
from src.carla.geometry.transforms import world_to_sensor
from src.common.typing_aliases import Float32Array, Float64Array


def build_projection_matrix(width: int, height: int, fov: float) -> Float32Array:
    """Build a pinhole camera intrinsics matrix from image size and field of view."""
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    matrix = np.identity(3, dtype=np.float32)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return np.asarray(matrix, dtype=np.float32)


def project_world_point(
    location: carla.Location,
    intrinsics: Float32Array,
    world_to_camera: Float32Array,
) -> Optional[Tuple[float, float, float]]:
    """Project one CARLA world point into image coordinates."""
    point = np.array([location.x, location.y, location.z, 1.0], dtype=np.float32)
    point_camera = world_to_camera @ point

    # Unreal Engine coordinates -> conventional camera coordinates.
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]], dtype=np.float32)
    if point_camera[2] <= 1e-3:
        return None

    point_image = intrinsics @ point_camera
    point_image[0] /= point_image[2]
    point_image[1] /= point_image[2]
    return float(point_image[0]), float(point_image[1]), float(point_camera[2])


def project_polyline_with_world_points(
    world_points: List[carla.Location],
    intrinsics: Float32Array,
    world_to_camera: Float32Array,
    image_width: int,
    image_height: int,
    projection_margin_px: float,
) -> List[Tuple[Float64Array, Float64Array]]:
    """Project a world-space polyline into image samples with paired 3D points."""
    projected_samples: List[Tuple[Float64Array, Float64Array]] = []

    for point in world_points:
        image_point = project_world_point(point, intrinsics, world_to_camera)
        if image_point is None:
            continue
        u, v, depth = image_point
        if depth <= 0:
            continue
        if (
            -projection_margin_px <= u < image_width + projection_margin_px
            and -projection_margin_px <= v < image_height + projection_margin_px
        ):
            projected_samples.append(
                (
                    np.asarray([float(u), float(v)], dtype=np.float64),
                    np.asarray([float(point.x), float(point.y), float(point.z)], dtype=np.float64),
                )
            )

    return projected_samples


def clip_line_segment_to_image_with_params(
    start: Float64Array,
    end: Float64Array,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[Float64Array, Float64Array, float, float]]:
    """Clip a projected line segment to the image rectangle while preserving interpolation."""
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx = x1 - x0
    dy = y1 - y0
    xmin, xmax = 0.0, float(image_width - 1)
    ymin, ymax = 0.0, float(image_height - 1)
    p = (-dx, dx, -dy, dy)
    q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u1, u2 = 0.0, 1.0

    for pi, qi in zip(p, q):
        if abs(pi) <= 1e-9:
            if qi < 0.0:
                return None
            continue

        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return None
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return None
            if t < u2:
                u2 = t

    clipped_start = np.asarray(start + (end - start) * u1, dtype=np.float64)
    clipped_end = np.asarray(start + (end - start) * u2, dtype=np.float64)
    return clipped_start, clipped_end, float(u1), float(u2)


def world_points_to_lidar(points_world: Float64Array, lidar_transform: carla.Transform) -> List[List[float]]:
    """Transform world-space lane points into the LiDAR frame."""
    if points_world.size == 0:
        return []

    points_lidar = world_to_sensor(points_world, lidar_transform)
    points_lidar[:, 1] *= -1.0
    return [[float(value) for value in row] for row in points_lidar.astype(np.float32)]


def clip_projected_polyline_to_image_and_lidar(
    projected_samples: List[Tuple[Float64Array, Float64Array]],
    image_width: int,
    image_height: int,
    lidar_transform: carla.Transform,
) -> Tuple[List[List[float]], List[List[float]], int]:
    """Clip projected lane samples and keep matching LiDAR-space points."""
    if len(projected_samples) < 2:
        return [], [], 0

    fragments_2d: List[List[Float64Array]] = []
    fragments_world: List[List[Float64Array]] = []
    current_fragment_2d: List[Float64Array] = []
    current_fragment_world: List[Float64Array] = []

    for (start_2d, start_world), (end_2d, end_world) in zip(projected_samples[:-1], projected_samples[1:]):
        clipped_segment = clip_line_segment_to_image_with_params(
            start_2d,
            end_2d,
            image_width,
            image_height,
        )
        if clipped_segment is None:
            if current_fragment_2d:
                fragments_2d.append(current_fragment_2d)
                fragments_world.append(current_fragment_world)
                current_fragment_2d = []
                current_fragment_world = []
            continue

        clipped_start_2d, clipped_end_2d, u1, u2 = clipped_segment
        clipped_start_world = start_world + (end_world - start_world) * u1
        clipped_end_world = start_world + (end_world - start_world) * u2

        if not current_fragment_2d:
            current_fragment_2d = [clipped_start_2d, clipped_end_2d]
            current_fragment_world = [clipped_start_world, clipped_end_world]
            continue

        if np.linalg.norm(current_fragment_2d[-1] - clipped_start_2d) > 1e-3:
            fragments_2d.append(current_fragment_2d)
            fragments_world.append(current_fragment_world)
            current_fragment_2d = [clipped_start_2d, clipped_end_2d]
            current_fragment_world = [clipped_start_world, clipped_end_world]
            continue

        current_fragment_2d.append(clipped_end_2d)
        current_fragment_world.append(clipped_end_world)

    if current_fragment_2d:
        fragments_2d.append(current_fragment_2d)
        fragments_world.append(current_fragment_world)

    if not fragments_2d:
        return [], [], 0

    best_index = max(range(len(fragments_2d)), key=lambda idx: len(fragments_2d[idx]))
    points_2d = [[float(value) for value in point.astype(np.float32)] for point in fragments_2d[best_index]]
    points_world = np.asarray(fragments_world[best_index], dtype=np.float64)
    points_lidar = world_points_to_lidar(points_world, lidar_transform)
    return points_2d, points_lidar, len(fragments_2d)


def dedupe_consecutive_points(points: List[List[float]], min_dist: float) -> List[List[float]]:
    """Remove nearly duplicated consecutive points from a polyline."""
    if not points:
        return points

    deduped = [points[0]]
    for point in points[1:]:
        dx = point[0] - deduped[-1][0]
        dy = point[1] - deduped[-1][1]
        if dx * dx + dy * dy >= min_dist * min_dist:
            deduped.append(point)
    return deduped
