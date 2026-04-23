from typing import Dict, List, Optional, Tuple

import numpy as np

import carla
from src.carla.geometry.transforms import world_to_sensor
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


def waypoint_key(waypoint: carla.Waypoint) -> Tuple[int, int, int, int]:
    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
        int(round(float(waypoint.s) * 100.0)),
    )


def choose_best_next_waypoint(current: carla.Waypoint, candidates: List[carla.Waypoint]) -> carla.Waypoint:
    if len(candidates) == 1:
        return candidates[0]

    current_forward = current.transform.get_forward_vector()

    def score(candidate: carla.Waypoint) -> Tuple[int, float]:
        candidate_forward = candidate.transform.get_forward_vector()
        alignment = (
            current_forward.x * candidate_forward.x
            + current_forward.y * candidate_forward.y
            + current_forward.z * candidate_forward.z
        )
        same_lane = int(
            candidate.road_id == current.road_id
            and candidate.section_id == current.section_id
            and candidate.lane_id == current.lane_id
        )
        return same_lane, float(alignment)

    return max(candidates, key=score)


def waypoint_chain_forward(start_waypoint: carla.Waypoint, distance_m: float, step_m: float) -> List[carla.Waypoint]:
    waypoints = [start_waypoint]
    traveled = 0.0
    current = start_waypoint
    seen = {waypoint_key(start_waypoint)}

    while traveled < distance_m:
        next_candidates = current.next(step_m)
        if not next_candidates:
            break
        current = choose_best_next_waypoint(current, next_candidates)
        current_key = waypoint_key(current)
        if current_key in seen:
            break
        seen.add(current_key)
        waypoints.append(current)
        traveled += step_m

    return waypoints


def _lane_side_sign(reference: carla.Waypoint, candidate: carla.Waypoint) -> float:
    delta_x = candidate.transform.location.x - reference.transform.location.x
    delta_y = candidate.transform.location.y - reference.transform.location.y
    delta_z = candidate.transform.location.z - reference.transform.location.z
    right = reference.transform.get_right_vector()
    return delta_x * right.x + delta_y * right.y + delta_z * right.z


def find_adjacent_driving_lane(
    carla_map: carla.Map,
    waypoint: carla.Waypoint,
    side: str,
    max_lane_id_delta: int = 8,
) -> Optional[carla.Waypoint]:
    adjacent = waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()
    if adjacent and adjacent.lane_type == carla.LaneType.Driving:
        return adjacent

    best_candidate: Optional[carla.Waypoint] = None
    best_distance = float("inf")

    for lane_id in range(-max_lane_id_delta, max_lane_id_delta + 1):
        if lane_id == 0 or lane_id == waypoint.lane_id:
            continue
        candidate = carla_map.get_waypoint_xodr(waypoint.road_id, lane_id, waypoint.s)
        if candidate is None:
            continue
        if candidate.section_id != waypoint.section_id:
            continue
        if candidate.lane_type != carla.LaneType.Driving:
            continue

        side_sign = _lane_side_sign(waypoint, candidate)
        if side == "left" and side_sign >= 0.0:
            continue
        if side == "right" and side_sign <= 0.0:
            continue

        lateral_distance = abs(float(side_sign))
        if lateral_distance < best_distance:
            best_distance = lateral_distance
            best_candidate = candidate

    return best_candidate


def collect_adjacent_driving_lanes(
    carla_map: carla.Map,
    base_waypoint: carla.Waypoint,
    max_side_lanes: int,
) -> List[carla.Waypoint]:
    lanes = [base_waypoint]

    current = base_waypoint
    for _ in range(max_side_lanes):
        left = find_adjacent_driving_lane(carla_map, current, side="left")
        if left and left.lane_type == carla.LaneType.Driving:
            lanes.append(left)
            current = left
        else:
            break

    current = base_waypoint
    for _ in range(max_side_lanes):
        right = find_adjacent_driving_lane(carla_map, current, side="right")
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


def boundary_key(carla_map: carla.Map, waypoint: carla.Waypoint, side: str) -> Tuple:
    if side == "left":
        neighbor = find_adjacent_driving_lane(carla_map, waypoint, side="left")
        if (
            neighbor
            and neighbor.lane_type == carla.LaneType.Driving
            and neighbor.road_id == waypoint.road_id
            and neighbor.section_id == waypoint.section_id
        ):
            lane_a, lane_b = sorted([waypoint.lane_id, neighbor.lane_id])
            return (waypoint.road_id, waypoint.section_id, lane_a, lane_b)
        return (waypoint.road_id, waypoint.section_id, waypoint.lane_id, "left_edge")

    neighbor = find_adjacent_driving_lane(carla_map, waypoint, side="right")
    if (
        neighbor
        and neighbor.lane_type == carla.LaneType.Driving
        and neighbor.road_id == waypoint.road_id
        and neighbor.section_id == waypoint.section_id
    ):
        lane_a, lane_b = sorted([waypoint.lane_id, neighbor.lane_id])
        return (waypoint.road_id, waypoint.section_id, lane_a, lane_b)
    return (waypoint.road_id, waypoint.section_id, waypoint.lane_id, "right_edge")


def sample_boundary_points(
    waypoint_chain: List[carla.Waypoint],
    side: str,
) -> Tuple[List[carla.Location], Dict]:
    points: List[carla.Location] = []
    metadata: Dict = {}
    has_non_none_marking = False

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
        marking_type = lane_marking_type_name(lane_marking.type)
        marking_color = lane_marking_color_name(lane_marking.color)
        if marking_type != "NONE":
            has_non_none_marking = True

        sample_metadata = {
            "road_id": waypoint.road_id,
            "section_id": waypoint.section_id,
            "lane_id": waypoint.lane_id,
            "side": side,
            "lane_width": float(waypoint.lane_width),
            "marking_type": marking_type,
            "marking_color": marking_color,
            "marking_width": float(lane_marking.width),
        }
        if not metadata or marking_type != "NONE":
            metadata = sample_metadata

    if metadata and not has_non_none_marking:
        metadata = {**metadata, "marking_type": "NONE"}

    return points, metadata


def project_polyline_with_world_points(
    world_points: List[carla.Location],
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    image_width: int,
    image_height: int,
    projection_margin_px: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    projected_samples: List[Tuple[np.ndarray, np.ndarray]] = []

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
    start: np.ndarray,
    end: np.ndarray,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
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

    clipped_start = start + (end - start) * u1
    clipped_end = start + (end - start) * u2
    return clipped_start, clipped_end, float(u1), float(u2)


def world_points_to_lidar(points_world: np.ndarray, lidar_transform: carla.Transform) -> List[List[float]]:
    if points_world.size == 0:
        return []

    points_lidar = world_to_sensor(points_world, lidar_transform)
    points_lidar[:, 1] *= -1.0
    return points_lidar.astype(np.float32).tolist()


def clip_projected_polyline_to_image_and_lidar(
    projected_samples: List[Tuple[np.ndarray, np.ndarray]],
    image_width: int,
    image_height: int,
    lidar_transform: carla.Transform,
) -> Tuple[List[List[float]], List[List[float]], int]:
    if len(projected_samples) < 2:
        return [], [], 0

    fragments_2d: List[List[np.ndarray]] = []
    fragments_world: List[List[np.ndarray]] = []
    current_fragment_2d: List[np.ndarray] = []
    current_fragment_world: List[np.ndarray] = []

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
    points_2d = [point.astype(np.float32).tolist() for point in fragments_2d[best_index]]
    points_world = np.asarray(fragments_world[best_index], dtype=np.float64)
    points_lidar = world_points_to_lidar(points_world, lidar_transform)
    return points_2d, points_lidar, len(fragments_2d)


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
    lidar_transform: carla.Transform,
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
    candidate_lanes = collect_adjacent_driving_lanes(
        carla_map,
        hero_waypoint,
        max_side_lanes=config.max_side_lanes,
    )

    annotations: List[Dict] = []
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
