from typing import Dict, List

import numpy as np

import carla
from src.carla.lanes.projection import (
    build_projection_matrix,
    clip_projected_polyline_to_image_and_lidar,
    dedupe_consecutive_points,
    project_polyline_with_world_points,
)
from src.carla.lanes.topology import (
    boundary_key,
    collect_adjacent_driving_lanes,
    sample_boundary_points,
    waypoint_chain_forward,
)
from src.common.config import LaneAnnotationsConfig
from src.common.typing_aliases import ObjectDict


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

            points_3d, metadata = sample_boundary_points(waypoint_chain, side)
            if metadata.get("marking_type") == "NONE":
                continue

            projected_samples = project_polyline_with_world_points(
                points_3d,
                intrinsics,
                world_to_camera,
                image_width,
                image_height,
                projection_margin_px=config.projection_margin_px,
            )
            clipped_points, clipped_points_lidar, visible_fragments = clip_projected_polyline_to_image_and_lidar(
                projected_samples,
                image_width=image_width,
                image_height=image_height,
                lidar_transform=lidar_transform,
            )
            if len(clipped_points) < 2:
                continue

            points_2d = dedupe_consecutive_points(clipped_points, min_dist=config.dedupe_distance_px)
            points_3d_lidar = dedupe_consecutive_points(clipped_points_lidar, min_dist=0.05)
            if len(points_2d) < 2:
                continue
            if len(points_3d_lidar) < 2:
                continue
            if visible_fragments < 1:
                continue

            annotations.append(
                {
                    "id": len(annotations),
                    **metadata,
                    "points": points_2d,
                    "points_lidar": points_3d_lidar,
                }
            )

    return annotations
