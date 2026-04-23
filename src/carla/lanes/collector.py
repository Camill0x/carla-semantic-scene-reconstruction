from typing import Dict, List, Optional, Tuple

import numpy as np

import carla
from src.common.config import LaneAnnotationsConfig


def build_projection_matrix(width: int, height: int, fov: float) -> np.ndarray:
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    matrix = np.identity(3, dtype=np.float32)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return matrix


def project_world_point(
    location: carla.Location,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
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


def waypoint_chain_forward(start_waypoint: carla.Waypoint, distance_m: float, step_m: float) -> List[carla.Waypoint]:
    waypoints = [start_waypoint]
    traveled = 0.0
    current = start_waypoint

    while traveled < distance_m:
        next_candidates = current.next(step_m)
        if not next_candidates:
            break
        current = next_candidates[0]
        waypoints.append(current)
        traveled += step_m

    return waypoints


def collect_adjacent_driving_lanes(base_waypoint: carla.Waypoint, max_side_lanes: int) -> List[carla.Waypoint]:
    lanes = [base_waypoint]

    current = base_waypoint
    for _ in range(max_side_lanes):
        left = current.get_left_lane()
        if left and left.lane_type == carla.LaneType.Driving:
            lanes.append(left)
            current = left
        else:
            break

    current = base_waypoint
    for _ in range(max_side_lanes):
        right = current.get_right_lane()
        if right and right.lane_type == carla.LaneType.Driving:
            lanes.append(right)
            current = right
        else:
            break

    unique = {}
    for waypoint in lanes:
        unique[(waypoint.road_id, waypoint.section_id, waypoint.lane_id)] = waypoint
    return list(unique.values())


def lane_marking_type_name(marking_type: carla.LaneMarkingType) -> str:
    return str(marking_type).split(".")[-1]


def lane_marking_color_name(color: carla.LaneMarkingColor) -> str:
    return str(color).split(".")[-1]


def boundary_key(waypoint: carla.Waypoint, side: str) -> Tuple:
    if side == "left":
        neighbor = waypoint.get_left_lane()
        if (
            neighbor
            and neighbor.lane_type == carla.LaneType.Driving
            and neighbor.road_id == waypoint.road_id
            and neighbor.section_id == waypoint.section_id
        ):
            lane_a, lane_b = sorted([waypoint.lane_id, neighbor.lane_id])
            return (waypoint.road_id, waypoint.section_id, lane_a, lane_b)
        return (waypoint.road_id, waypoint.section_id, waypoint.lane_id, "left_edge")

    neighbor = waypoint.get_right_lane()
    if (
        neighbor
        and neighbor.lane_type == carla.LaneType.Driving
        and neighbor.road_id == waypoint.road_id
        and neighbor.section_id == waypoint.section_id
    ):
        lane_a, lane_b = sorted([waypoint.lane_id, neighbor.lane_id])
        return (waypoint.road_id, waypoint.section_id, lane_a, lane_b)
    return (waypoint.road_id, waypoint.section_id, waypoint.lane_id, "right_edge")


def sample_boundary_points(waypoint_chain: List[carla.Waypoint], side: str) -> Tuple[List[carla.Location], Dict]:
    points: List[carla.Location] = []
    metadata: Dict = {}

    for waypoint in waypoint_chain:
        transform = waypoint.transform
        location = transform.location
        right_vector = transform.get_right_vector()
        half_width = 0.5 * waypoint.lane_width

        if side == "left":
            lane_marking = waypoint.left_lane_marking
            point = carla.Location(
                x=location.x - right_vector.x * half_width,
                y=location.y - right_vector.y * half_width,
                z=location.z + 0.05,
            )
        else:
            lane_marking = waypoint.right_lane_marking
            point = carla.Location(
                x=location.x + right_vector.x * half_width,
                y=location.y + right_vector.y * half_width,
                z=location.z + 0.05,
            )

        points.append(point)
        metadata = {
            "road_id": waypoint.road_id,
            "section_id": waypoint.section_id,
            "lane_id": waypoint.lane_id,
            "side": side,
            "lane_width": float(waypoint.lane_width),
            "marking_type": lane_marking_type_name(lane_marking.type),
            "marking_color": lane_marking_color_name(lane_marking.color),
            "marking_width": float(lane_marking.width),
        }

    return points, metadata


def project_polyline(
    world_points: List[carla.Location],
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    image_width: int,
    image_height: int,
    projection_margin_px: float,
) -> List[List[float]]:
    projected_points: List[List[float]] = []

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
            projected_points.append([float(u), float(v)])

    return projected_points


def dedupe_consecutive_points(points: List[List[float]], min_dist: float) -> List[List[float]]:
    if not points:
        return points

    deduped = [points[0]]
    for point in points[1:]:
        dx = point[0] - deduped[-1][0]
        dy = point[1] - deduped[-1][1]
        if dx * dx + dy * dy >= min_dist * min_dist:
            deduped.append(point)
    return deduped


def collect_lane_annotations(
    world: carla.World,
    hero: carla.Actor,
    camera_transform: carla.Transform,
    image_width: int,
    image_height: int,
    camera_fov: float,
    config: LaneAnnotationsConfig,
) -> List[Dict]:
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
    candidate_lanes = collect_adjacent_driving_lanes(hero_waypoint, max_side_lanes=config.max_side_lanes)

    annotations: List[Dict] = []
    used_keys = set()

    for lane_waypoint in candidate_lanes:
        waypoint_chain = waypoint_chain_forward(
            lane_waypoint,
            distance_m=config.distance_m,
            step_m=config.step_m,
        )

        for side in ("left", "right"):
            key = boundary_key(lane_waypoint, side)
            if key in used_keys:
                continue
            used_keys.add(key)

            points_3d, metadata = sample_boundary_points(waypoint_chain, side)
            if metadata.get("marking_type") == "NONE":
                continue

            points_2d = project_polyline(
                points_3d,
                intrinsics,
                world_to_camera,
                image_width,
                image_height,
                projection_margin_px=config.projection_margin_px,
            )
            points_2d = dedupe_consecutive_points(points_2d, min_dist=config.dedupe_distance_px)
            if len(points_2d) < 2:
                continue

            annotations.append(
                {
                    "id": len(annotations),
                    **metadata,
                    "points": points_2d,
                }
            )

    return annotations
