from typing import List, Optional, Sequence, Tuple

import numpy as np

import carla
from src.carla.lanes.projection import (
    build_projection_matrix,
    clip_projected_polyline_fragments_to_image_and_lidar,
    project_polyline_with_world_points,
)
from src.carla.lanes.topology import (
    boundary_key,
    collect_adjacent_driving_lanes,
    sample_boundary_segments,
    waypoint_chain_forward,
)
from src.common.config import LaneAnnotationsConfig
from src.common.typing_aliases import ObjectDict


def bottommost_y(points: Sequence[Sequence[float]]) -> float:
    """Return the largest image-space y coordinate in a polyline."""
    if not points:
        return float("-inf")
    return max(float(point[1]) for point in points)


def extend_polyline_to_bottom(
    points_2d: List[List[float]],
    points_3d_lidar: List[List[float]],
    *,
    image_height: int,
    threshold_px: float,
) -> tuple[List[List[float]], List[List[float]]]:
    """Extend a lane polyline to the bottom image edge when it already starts close to the ego side."""
    if len(points_2d) < 2 or len(points_3d_lidar) < 2:
        return points_2d, points_3d_lidar

    first = points_2d[0]
    second = points_2d[1]
    bottom_y = float(image_height - 1)
    if bottom_y - float(first[1]) > threshold_px:
        return points_2d, points_3d_lidar

    dy = float(first[1] - second[1])
    if dy <= 1e-3:
        return points_2d, points_3d_lidar

    extension = (bottom_y - float(first[1])) / dy
    if extension <= 0.0 or extension > 3.0:
        return points_2d, points_3d_lidar

    dx = float(first[0] - second[0])
    new_point_2d = [
        float(first[0] + dx * extension),
        bottom_y,
    ]

    first_3d = np.asarray(points_3d_lidar[0], dtype=np.float64)
    second_3d = np.asarray(points_3d_lidar[1], dtype=np.float64)
    new_point_3d = (first_3d + (first_3d - second_3d) * extension).astype(np.float32)

    return [new_point_2d] + points_2d, [[float(value) for value in new_point_3d]] + points_3d_lidar


def dedupe_and_measure_polyline(
    points_2d: Sequence[Sequence[float]],
    points_3d_lidar: Sequence[Sequence[float]],
    *,
    min_dist_2d: float,
    min_dist_3d: float,
) -> Tuple[List[List[float]], List[List[float]], float, float]:
    """Dedupe 2D and 3D polylines and measure their cumulative lengths in one pass each."""
    if not points_2d or not points_3d_lidar:
        return [], [], 0.0, 0.0

    deduped_2d: List[List[float]] = [[float(value) for value in points_2d[0]]]
    deduped_3d: List[List[float]] = [[float(value) for value in points_3d_lidar[0]]]
    length_2d = 0.0
    length_3d = 0.0
    min_dist_2d_sq = float(min_dist_2d * min_dist_2d)
    min_dist_3d_sq = float(min_dist_3d * min_dist_3d)

    for point in points_2d[1:]:
        dx = float(point[0]) - deduped_2d[-1][0]
        dy = float(point[1]) - deduped_2d[-1][1]
        dist_sq = dx * dx + dy * dy
        if dist_sq >= min_dist_2d_sq:
            deduped_2d.append([float(point[0]), float(point[1])])
            length_2d += float(np.sqrt(dist_sq))

    for point in points_3d_lidar[1:]:
        dx = float(point[0]) - deduped_3d[-1][0]
        dy = float(point[1]) - deduped_3d[-1][1]
        dz = float(point[2]) - deduped_3d[-1][2]
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq >= min_dist_3d_sq:
            deduped_3d.append([float(point[0]), float(point[1]), float(point[2])])
            length_3d += float(np.sqrt(dist_sq))

    return deduped_2d, deduped_3d, length_2d, length_3d


def collect_lane_annotations(
    world: carla.World,
    hero: carla.Actor,
    lidar_transform: carla.Transform,
    camera_transform: carla.Transform,
    image_width: int,
    image_height: int,
    camera_fov: float,
    config: LaneAnnotationsConfig,
) -> List[ObjectDict]:
    """Collect lane annotations around the ego vehicle and project them into 2D and 3D."""
    carla_map = world.get_map()
    hero_waypoint = carla_map.get_waypoint(
        hero.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if hero_waypoint is None:
        return []

    world_to_camera = np.array(camera_transform.get_inverse_matrix(), dtype=np.float32)
    intrinsics = build_projection_matrix(image_width, image_height, camera_fov)
    candidate_lanes = collect_adjacent_driving_lanes(
        carla_map,
        hero_waypoint,
        max_side_lanes=config.max_side_lanes,
    )

    annotations: List[ObjectDict] = []
    used_keys = set()

    for lane_waypoint in candidate_lanes:
        waypoint_chain = waypoint_chain_forward(
            lane_waypoint,
            distance_m=config.distance_m,
            step_m=config.step_m,
        )

        for side in ("left", "right"):
            key = boundary_key(carla_map, lane_waypoint, side)
            if key in used_keys:
                continue
            used_keys.add(key)

            boundary_segments = sample_boundary_segments(
                waypoint_chain,
                side,
                min_segment_points=config.min_segment_points,
            )
            best_fragment: Optional[ObjectDict] = None
            best_fragment_y = float("-inf")
            for points_3d, metadata in boundary_segments:
                projected_samples = project_polyline_with_world_points(
                    points_3d,
                    intrinsics,
                    world_to_camera,
                    image_width,
                    image_height,
                    projection_margin_px=config.projection_margin_px,
                )
                clipped_fragments = clip_projected_polyline_fragments_to_image_and_lidar(
                    projected_samples,
                    image_width=image_width,
                    image_height=image_height,
                    lidar_transform=lidar_transform,
                )
                for clipped_points, clipped_points_lidar in clipped_fragments:
                    bottom_y = bottommost_y(clipped_points)
                    if bottom_y <= best_fragment_y:
                        continue
                    best_fragment_y = bottom_y
                    best_fragment = {
                        **metadata,
                        "points": clipped_points,
                        "points_lidar": clipped_points_lidar,
                    }

            if best_fragment is None:
                continue

            points_2d, points_3d_lidar = extend_polyline_to_bottom(
                best_fragment["points"],
                best_fragment["points_lidar"],
                image_height=image_height,
                threshold_px=config.extend_to_bottom_threshold_px,
            )
            points_2d, points_3d_lidar, length_2d, length_3d = dedupe_and_measure_polyline(
                points_2d,
                points_3d_lidar,
                min_dist_2d=config.dedupe_distance_px,
                min_dist_3d=0.05,
            )
            if len(points_2d) < config.min_projected_points:
                continue
            if len(points_3d_lidar) < config.min_projected_points:
                continue
            if length_2d < config.min_length_px:
                continue
            if length_3d < config.min_length_m:
                continue

            annotations.append(
                {
                    "id": len(annotations),
                    **{key: value for key, value in best_fragment.items() if key not in {"points", "points_lidar"}},
                    "points": points_2d,
                    "points_lidar": points_3d_lidar,
                }
            )

    return annotations
