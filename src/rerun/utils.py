from typing import List, Sequence, Tuple

import numpy as np

from rerun.datatypes import Angle, RotationAxisAngle
from src.common.typing_aliases import Float32Array, JsonDict
from src.openpcdet.prediction import Objects3DPrediction

EMPTY_POINTS: Float32Array = np.zeros((0, 3), dtype=np.float32)


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
