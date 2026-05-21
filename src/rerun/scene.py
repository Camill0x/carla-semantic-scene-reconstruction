from typing import Sequence, Union

import numpy as np

import rerun as rr
from src.common.typing_aliases import Float32Array, ImageArray, JsonDict
from src.lanedet.prediction import Lanes2DPrediction, Lanes3DPrediction
from src.openpcdet.prediction import Objects3DPrediction
from src.rerun.colors import EGO_COLOR, GT_COLOR, lane_color, point_colors, prediction_colors
from src.rerun.utils import (
    EMPTY_POINTS,
    boxes_to_rerun,
    gt_labels,
    point_positions,
    prediction_labels,
    prediction_names,
    prediction_scores,
)


def clear_entity(path: str) -> None:
    """Clear a logged Rerun entity when the backend supports it."""
    clear = getattr(rr, "Clear", None)
    if clear is not None:
        rr.log(path, clear(recursive=False))
        return


def log_camera_front_image(image_rgb: ImageArray) -> None:
    """Log the front camera image to the Rerun image view."""
    rr.log("camera/front/image", rr.Image(image_rgb))


def log_points(points: Float32Array, *, point_radius: float, visible: bool) -> None:
    """Log a point cloud to the Rerun 3D scene."""
    if not visible:
        rr.log("world/points", rr.Points3D(positions=EMPTY_POINTS))
        return

    rr.log(
        "world/points",
        rr.Points3D(
            positions=point_positions(points),
            colors=point_colors(points),
            radii=point_radius,
        ),
    )


def log_gt_boxes(
    gt_boxes: Float32Array,
    gt_names: Sequence[str],
    *,
    line_radius: float,
    visible: bool,
) -> None:
    """Log ground-truth 3D boxes to the Rerun 3D scene."""
    if not visible:
        rr.log("world/objects_GT", rr.Boxes3D(centers=np.zeros((0, 3), dtype=np.float32)))
        return

    centers, half_sizes, rotations = boxes_to_rerun(gt_boxes)
    rr.log(
        "world/objects_GT",
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            rotation_axis_angles=rotations,
            colors=[GT_COLOR for _ in range(len(centers))],
            labels=gt_labels(gt_boxes, gt_names),
            show_labels=True,
            radii=line_radius,
            fill_mode="solid",
        ),
    )


def log_ego_box(
    ego_box: Float32Array,
    *,
    line_radius: float,
) -> None:
    """Log the ego-vehicle box to the Rerun 3D scene."""
    if ego_box.size == 0:
        rr.log("world/ego", rr.Boxes3D(centers=np.zeros((0, 3), dtype=np.float32)))
        return

    ego_boxes = np.asarray(ego_box, dtype=np.float32).reshape(1, 7)
    centers, half_sizes, rotations = boxes_to_rerun(ego_boxes)
    rr.log(
        "world/ego",
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            rotation_axis_angles=rotations,
            colors=[EGO_COLOR],
            labels=["Ego"],
            show_labels=True,
            radii=line_radius,
            fill_mode="solid",
        ),
    )


def log_prediction_objects_3d(
    objects_3d: Objects3DPrediction,
    *,
    line_radius: float,
) -> None:
    """Log predicted 3D objects to the Rerun 3D scene."""
    centers, half_sizes, rotations = boxes_to_rerun(objects_3d.boxes)
    if len(centers) == 0:
        clear_entity("world/objects_predicted")
        return

    names = prediction_names(objects_3d)
    labels = prediction_labels(names)
    colors = prediction_colors(names)
    scores = prediction_scores(objects_3d.scores)
    rr.log(
        "world/objects_predicted",
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            rotation_axis_angles=rotations,
            colors=colors,
            labels=labels,
            show_labels=True,
            radii=line_radius,
            fill_mode="solid",
        ),
        rr.AnyValues(score=scores),
    )


def log_lane_annotations_2d(lanes: Sequence[JsonDict], *, line_thickness: float) -> None:
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
        rr.log("camera/front/lanes_GT", rr.LineStrips2D(strips=[]))
        return

    rr.log(
        "camera/front/lanes_GT",
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
        rr.log("world/lanes_GT", rr.LineStrips3D(strips=[]))
        return

    rr.log(
        "world/lanes_GT",
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

        strips.append(points)
        colors.append((255, 255, 0, 235))
        labels.append(name)
        scores_log.append(round(float(score), 2) if score is not None else None)

    if not strips:
        clear_entity("camera/front/lanes_predicted")
        return

    rr.log(
        "camera/front/lanes_predicted",
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

        strips.append(points)
        colors.append((255, 255, 255, 235))
        labels.append(name)
        scores_log.append(round(float(score), 2) if score is not None else None)

    if not strips:
        clear_entity("world/lanes_predicted")
        return

    rr.log(
        "world/lanes_predicted",
        rr.LineStrips3D(
            strips=strips,
            colors=colors,
            radii=line_radius,
            labels=labels,
            show_labels=False,
        ),
        rr.AnyValues(score=scores_log),
    )
