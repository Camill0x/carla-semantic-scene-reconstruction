from typing import List, Sequence, Tuple

import numpy as np

import rerun as rr
from rerun.datatypes import Angle, RotationAxisAngle
from src.common.typing_aliases import Float32Array, ImageArray
from src.openpcdet.prediction import Objects3DPrediction
from src.rerun.colors import EGO_COLOR, GT_COLOR, prediction_colors

EMPTY_POINTS: Float32Array = np.zeros((0, 3), dtype=np.float32)
EMPTY_COLORS: ImageArray = np.zeros((0, 4), dtype=np.uint8)


def clear_entity(path: str) -> None:
    """Clear a logged Rerun entity when the backend supports it."""
    clear = getattr(rr, "Clear", None)
    if clear is not None:
        rr.log(path, clear(recursive=False))
        return


def boxes_to_rerun(boxes: Float32Array) -> Tuple[Float32Array, Float32Array, List[RotationAxisAngle]]:
    """Convert 7D boxes into the centers, sizes, and rotations expected by Rerun."""
    if boxes.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            [],
        )

    centers = boxes[:, 0:3]
    half_sizes = boxes[:, 3:6] * 0.5
    rotations = [RotationAxisAngle(axis=[0.0, 0.0, 1.0], angle=Angle(rad=float(yaw))) for yaw in boxes[:, 6]]
    return centers, half_sizes, rotations


def gt_labels(gt_boxes: Float32Array, gt_names: Sequence[str]) -> List[str]:
    """Build display labels for ground-truth boxes."""
    if gt_names and len(gt_names) == len(gt_boxes):
        return [f"GT {name}" for name in gt_names]
    return [f"GT #{idx}" for idx in range(len(gt_boxes))]


def prediction_labels(pred_names: Sequence[str]) -> List[str]:
    """Build display labels for predicted objects."""
    return [str(name) for name in pred_names]


def prediction_names(objects_3d: Objects3DPrediction) -> List[str]:
    """Return prediction class names aligned with the predicted boxes."""
    if len(objects_3d.names) == len(objects_3d.boxes):
        return [str(name) for name in objects_3d.names]
    if len(objects_3d.labels) == len(objects_3d.boxes):
        return [f"label_{int(label)}" for label in objects_3d.labels]
    return [f"object_{index}" for index in range(len(objects_3d.boxes))]


def prediction_scores(scores: Float32Array) -> List[float]:
    """Return rounded prediction scores for display."""
    return [round(float(value), 2) for value in scores]


def point_positions(points: Float32Array) -> Float32Array:
    """Extract 3D point positions from a point cloud array."""
    if points.size == 0:
        return EMPTY_POINTS
    return points[:, :3]


def point_colors(points: Float32Array) -> ImageArray:
    """Build per-point RGB colors from LiDAR intensity values."""
    if points.size == 0:
        return EMPTY_COLORS

    if points.shape[1] >= 4:
        intensity = points[:, 3]
        if intensity.size > 0 and float(intensity.max()) > float(intensity.min()):
            normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        else:
            normalized = np.full((points.shape[0],), 0.7, dtype=np.float32)
    else:
        normalized = np.full((points.shape[0],), 0.7, dtype=np.float32)

    shade = (170 + 85 * normalized).astype(np.uint8)
    blue = np.full((points.shape[0],), 255, dtype=np.uint8)
    alpha = np.full((points.shape[0],), 220, dtype=np.uint8)
    return np.stack([shade, shade, blue, alpha], axis=1)


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
