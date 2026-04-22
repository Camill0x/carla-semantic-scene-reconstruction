from typing import Iterable, Sequence, Tuple

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.datatypes import Angle, RotationAxisAngle

CLASS_COLORS = {
    "car": (242, 51, 51, 235),
    "truck": (191, 26, 26, 235),
    "bus": (255, 115, 64, 235),
    "motorcycle": (255, 179, 26, 235),
    "bicycle": (242, 230, 51, 235),
    "pedestrian": (38, 217, 64, 235),
}

GT_COLOR = (77, 163, 255, 210)
EMPTY_POINTS = np.zeros((0, 3), dtype=np.float32)
EMPTY_COLORS = np.zeros((0, 4), dtype=np.uint8)


def initialize_viewer(application_id: str, *, show_grid: bool) -> None:
    rr.init(application_id, spawn=True)
    rr.send_blueprint(make_blueprint(show_grid=show_grid))
    rr.log("world", rr.ViewCoordinates.FLU, static=True)


def make_blueprint(show_grid: bool):
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="Live Scene",
                line_grid=rrb.LineGrid3D(visible=show_grid),
                eye_controls=rrb.EyeControls3D(
                    position=(-22.0, 0.0, 10.5),
                    look_target=(8.0, 0.0, 0.0),
                    eye_up=(0.0, 0.0, 1.0),
                    speed=18.0,
                ),
            ),
            rrb.Vertical(
                rrb.TextDocumentView(origin="/status", name="Status"),
                rrb.TextDocumentView(origin="/legend", name="Legend"),
                row_shares=[0.65, 0.35],
            ),
            column_shares=[0.82, 0.18],
        ),
        collapse_panels=False,
    )


def log_status(
    *,
    frame: int,
    num_points: int,
    num_gt: int,
    num_pred: int,
    score_thresh: float | None = None,
) -> None:
    lines = [
        f"Frame: {frame}",
        f"Points: {num_points}",
        f"GT boxes: {num_gt}",
        f"Pred boxes: {num_pred}",
    ]
    if score_thresh is not None:
        lines.append(f"Score threshold: {score_thresh:.2f}")
    rr.log("status", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN))


def log_legend() -> None:
    lines = [
        "# Colors",
        "- GT boxes: blue",
        "- Car: red",
        "- Truck: dark red",
        "- Bus: orange",
        "- Motorcycle: amber",
        "- Bicycle: yellow",
        "- Pedestrian: green",
        "- Points: intensity shaded",
    ]
    rr.log("legend", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)


def boxes_to_rerun(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
    if boxes.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            [],
        )

    centers = boxes[:, 0:3].astype(np.float32)
    half_sizes = (boxes[:, 3:6] * 0.5).astype(np.float32)
    rotations = [
        RotationAxisAngle(axis=[0.0, 0.0, 1.0], angle=Angle(rad=float(yaw)))
        for yaw in boxes[:, 6]
    ]
    return centers, half_sizes, rotations


def prediction_colors(pred_names: Iterable[str]) -> list:
    return [CLASS_COLORS.get(str(name), (255, 255, 255, 235)) for name in pred_names]


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


def log_prediction_boxes(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_names: Sequence[str],
    *,
    line_radius: float,
) -> None:
    centers, half_sizes, rotations = boxes_to_rerun(pred_boxes)
    rr.log(
        "world/predictions",
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            rotation_axis_angles=rotations,
            colors=prediction_colors(pred_names),
            labels=prediction_labels(pred_names, pred_scores),
            show_labels=True,
            radii=line_radius,
            fill_mode="solid",
        ),
    )
