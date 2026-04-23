from typing import List, Sequence

import numpy as np

import rerun as rr
from src.rerun.colors import lane_color


def clamp_lane_annotations_to_image(lanes: List[dict], image_width: int, image_height: int) -> List[dict]:
    clamped_lanes: List[dict] = []
    max_x = float(max(image_width - 1, 0))
    max_y = float(max(image_height - 1, 0))

    for lane in lanes:
        points = lane.get("points", [])
        if len(points) < 2:
            continue

        clamped_points = []
        for point in points:
            if len(point) != 2:
                continue
            clamped_points.append(
                [
                    float(np.clip(float(point[0]), 0.0, max_x)),
                    float(np.clip(float(point[1]), 0.0, max_y)),
                ]
            )

        if len(clamped_points) < 2:
            continue

        clamped_lanes.append({**lane, "points": clamped_points})

    return clamped_lanes


def log_lane_annotations_2d(lanes: List[dict], *, line_thickness: float) -> None:
    strips = []
    colors = []
    labels = []

    for lane in lanes:
        points = lane.get("points", [])
        if len(points) < 2:
            continue
        strips.append(np.asarray(points, dtype=np.float32))
        colors.append(lane_color(lane))
        labels.append(f"lane {lane.get('lane_id', '?')} {lane.get('side', '?')} {lane.get('marking_type', '?')}")

    if not strips:
        rr.log("camera/front/lanes", rr.LineStrips2D(strips=[]))
        return

    rr.log(
        "camera/front/lanes",
        rr.LineStrips2D(
            strips=strips,
            colors=colors,
            radii=line_thickness,
            labels=labels,
            show_labels=False,
        ),
    )


def log_lane_annotations_3d(
    lanes: Sequence[dict],
    *,
    line_radius: float,
) -> None:
    strips = []
    colors = []
    labels = []

    for lane in lanes:
        points_lidar = lane.get("points_lidar", [])
        if len(points_lidar) < 2:
            continue
        strips.append(np.asarray(points_lidar, dtype=np.float32))
        colors.append(lane_color(lane))
        labels.append(f"lane {lane.get('lane_id', '?')} {lane.get('side', '?')} {lane.get('marking_type', '?')}")

    if not strips:
        rr.log("world/lanes", rr.LineStrips3D(strips=[]))
        return

    rr.log(
        "world/lanes",
        rr.LineStrips3D(
            strips=strips,
            colors=colors,
            radii=line_radius,
            labels=labels,
            show_labels=False,
        ),
    )
