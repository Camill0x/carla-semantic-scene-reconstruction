from typing import List, Optional, Tuple

import carla
from src.common.typing_aliases import BoundaryKey, BoundaryMetadata


def waypoint_key(waypoint: carla.Waypoint) -> Tuple[int, int, int, int]:
    """Build a stable hashable key for a CARLA waypoint."""
    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
        int(round(float(waypoint.s) * 100.0)),
    )


def choose_best_next_waypoint(current: carla.Waypoint, candidates: List[carla.Waypoint]) -> carla.Waypoint:
    """Choose the forward waypoint candidate that best continues the current lane."""
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
    """Trace a waypoint chain forward for the requested distance."""
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


def lane_side_sign(reference: carla.Waypoint, candidate: carla.Waypoint) -> float:
    """Return the signed lateral side of a candidate lane relative to a reference waypoint."""
    delta_x = candidate.transform.location.x - reference.transform.location.x
    delta_y = candidate.transform.location.y - reference.transform.location.y
    delta_z = candidate.transform.location.z - reference.transform.location.z
    right = reference.transform.get_right_vector()
    return float(delta_x * right.x + delta_y * right.y + delta_z * right.z)


def find_adjacent_driving_lane(
    carla_map: carla.Map,
    waypoint: carla.Waypoint,
    side: str,
    max_lane_id_delta: int = 8,
) -> Optional[carla.Waypoint]:
    """Find the nearest adjacent driving lane on the requested side."""
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

        side_sign = lane_side_sign(waypoint, candidate)
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
    """Collect adjacent driving lanes around the base waypoint on both sides."""
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
    """Return the enum name of a CARLA lane-marking type."""
    return str(marking_type).split(".")[-1]


def lane_marking_color_name(color: carla.LaneMarkingColor) -> str:
    """Return the enum name of a CARLA lane-marking color."""
    return str(color).split(".")[-1]


def is_visible_lane_marking_type(marking_type: str) -> bool:
    """Return whether a CARLA lane-marking type represents a painted lane line worth annotating."""
    return marking_type not in {"NONE", "Other", "Grass", "Curb"}


def boundary_key(carla_map: carla.Map, waypoint: carla.Waypoint, side: str) -> BoundaryKey:
    """Build a stable identifier for one lane boundary relative to its neighbors."""
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


def sample_boundary_segments(
    waypoint_chain: List[carla.Waypoint],
    side: str,
    min_segment_points: int,
) -> List[Tuple[List[carla.Location], BoundaryMetadata]]:
    """Sample contiguous visible boundary segments from a waypoint chain."""
    segments: List[Tuple[List[carla.Location], BoundaryMetadata]] = []
    current_points: List[carla.Location] = []
    current_metadata: BoundaryMetadata = {}

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

        marking_type = lane_marking_type_name(lane_marking.type)
        marking_color = lane_marking_color_name(lane_marking.color)

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

        if not is_visible_lane_marking_type(marking_type):
            if len(current_points) >= min_segment_points and current_metadata:
                segments.append((current_points, current_metadata))
            current_points = []
            current_metadata = {}
            continue

        current_points.append(point)
        if not current_metadata:
            current_metadata = sample_metadata

    if len(current_points) >= min_segment_points and current_metadata:
        segments.append((current_points, current_metadata))

    return segments
