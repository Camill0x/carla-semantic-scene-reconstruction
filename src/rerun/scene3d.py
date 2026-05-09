from typing import Sequence, Tuple

import numpy as np

import rerun as rr
from rerun.datatypes import Angle, RotationAxisAngle
from src.openpcdet.prediction import Objects3DPrediction
from src.rerun.colors import EGO_COLOR, GT_COLOR, prediction_colors

EMPTY_POINTS = np.zeros((0, 3), dtype=np.float32)
EMPTY_COLORS = np.zeros((0, 4), dtype=np.uint8)


def boxes_to_rerun(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
    if boxes.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            [],
        )

    centers = boxes[:, 0:3].astype(np.float32)
    half_sizes = (boxes[:, 3:6] * 0.5).astype(np.float32)
    rotations = [RotationAxisAngle(axis=[0.0, 0.0, 1.0], angle=Angle(rad=float(yaw))) for yaw in boxes[:, 6]]
    return centers, half_sizes, rotations


def gt_labels(gt_boxes: np.ndarray, gt_names: Sequence[str]) -> list:
    if gt_names and len(gt_names) == len(gt_boxes):
        return [f"GT {name}" for name in gt_names]
    return [f"GT #{idx}" for idx in range(len(gt_boxes))]


def prediction_labels(pred_names: Sequence[str], pred_scores: np.ndarray) -> list:
    return [f"{name}  {float(score):.2f}" for name, score in zip(pred_names, pred_scores)]


def point_positions(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return EMPTY_POINTS
    return points[:, :3].astype(np.float32)


def point_colors(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return EMPTY_COLORS

    if points.shape[1] >= 4:
        intensity = points[:, 3].astype(np.float32)
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


def log_points(points: np.ndarray, *, point_radius: float, visible: bool) -> None:
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
    gt_boxes: np.ndarray,
    gt_names: Sequence[str],
    *,
    line_radius: float,
    visible: bool,
) -> None:
    if not visible:
        rr.log("world/gt", rr.Boxes3D(centers=np.zeros((0, 3), dtype=np.float32)))
        return

    centers, half_sizes, rotations = boxes_to_rerun(gt_boxes)
    rr.log(
        "world/gt",
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
    ego_box: np.ndarray,
    *,
    line_radius: float,
) -> None:
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
    centers, half_sizes, rotations = boxes_to_rerun(objects_3d.boxes)
    rr.log(
        "world/predictions",
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            rotation_axis_angles=rotations,
            colors=prediction_colors(objects_3d.names),
            labels=prediction_labels(objects_3d.names, objects_3d.scores),
            show_labels=True,
            radii=line_radius,
            fill_mode="solid",
        ),
    )
