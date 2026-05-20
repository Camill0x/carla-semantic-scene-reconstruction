from typing import List, Sequence, Union

import numpy as np

import rerun as rr
from src.common.typing_aliases import JsonDict
from src.lanedet.prediction import Lanes2DPrediction, Lanes3DPrediction
from src.rerun.colors import lane_color
from src.rerun.scene3d import clear_entity


def clamp_lane_annotations_to_image(
    lanes: List[JsonDict],
    image_width: int,
    image_height: int,
) -> List[JsonDict]:
    """Clamp lane-annotation points to the image bounds."""
    clamped_lanes: List[JsonDict] = []
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


def log_lane_annotations_2d(lanes: List[JsonDict], *, line_thickness: float) -> None:
    """Log collected 2D lane annotations to the Rerun image view."""
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
    lanes: Sequence[JsonDict],
    *,
    line_radius: float,
) -> None:
    """Log collected 3D lane annotations to the Rerun 3D scene."""
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


def log_prediction_lanes_2d(lanes_2d: Union[Lanes2DPrediction, JsonDict], *, line_thickness: float) -> None:
    """Log predicted 2D lanes to the Rerun image view."""
    if isinstance(lanes_2d, Lanes2DPrediction):
        strips_payload = lanes_2d.strips
        scores = lanes_2d.scores
        names = lanes_2d.names
    else:
        strips_payload = lanes_2d.get("strips", []) if isinstance(lanes_2d, dict) else []
        scores = lanes_2d.get("scores", []) if isinstance(lanes_2d, dict) else []
        names = lanes_2d.get("names", []) if isinstance(lanes_2d, dict) else []

    strips = []
    colors = []
    labels = []
    scores_log = []

    for index, strip in enumerate(strips_payload):
        points = np.asarray(strip, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
            continue

        name = str(names[index]) if index < len(names) else f"lane {index}"
        score = scores[index] if index < len(scores) else None
        label = name

        strips.append(points)
        colors.append((255, 255, 0, 235))
        labels.append(label)
        scores_log.append(round(float(score), 2) if score is not None else None)

    if not strips:
        rr.log("camera/front/predicted_lanes", rr.LineStrips2D(strips=[]))
        return

    rr.log(
        "camera/front/predicted_lanes",
        rr.LineStrips2D(
            strips=strips,
            colors=colors,
            radii=line_thickness,
            labels=labels,
            show_labels=False,
        ),
        rr.AnyValues(score=scores_log),
    )


def log_prediction_lanes_3d(lanes_3d: Union[Lanes3DPrediction, JsonDict], *, line_radius: float) -> None:
    """Log predicted 3D lanes to the Rerun 3D scene."""
    if isinstance(lanes_3d, Lanes3DPrediction):
        strips_payload = lanes_3d.strips
        scores = lanes_3d.scores
        names = lanes_3d.names
    else:
        strips_payload = lanes_3d.get("strips", []) if isinstance(lanes_3d, dict) else []
        scores = lanes_3d.get("scores", []) if isinstance(lanes_3d, dict) else []
        names = lanes_3d.get("names", []) if isinstance(lanes_3d, dict) else []

    strips = []
    colors = []
    labels = []
    scores_log = []

    for index, strip in enumerate(strips_payload):
        points = np.asarray(strip, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            continue

        name = str(names[index]) if index < len(names) else f"lane {index}"
        score = scores[index] if index < len(scores) else None
        label = name

        strips.append(points)
        colors.append((255, 255, 255, 235))
        labels.append(label)
        scores_log.append(round(float(score), 2) if score is not None else None)

    if not strips:
        clear_entity("world/predicted_lanes")
        return

    rr.log(
        "world/predicted_lanes",
        rr.LineStrips3D(
            strips=strips,
            colors=colors,
            radii=line_radius,
            labels=labels,
            show_labels=False,
        ),
        rr.AnyValues(score=scores_log),
    )
