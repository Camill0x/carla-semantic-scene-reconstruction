from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


def empty_gt_payload() -> Dict[str, Any]:
    return {
        "num_objects": 0,
        "class_counts": {},
        "gt_names": [],
        "objects": [],
        "gt_boxes": np.zeros((0, 7), dtype=np.float32),
    }


def build_lidar_message(
    *,
    frame: int,
    timestamp: float,
    max_range: float,
    classes: Sequence[str],
    hero_id: int,
    hero_type_id: str,
    points: np.ndarray,
    gt_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    message = {
        "frame": int(frame),
        "timestamp": float(timestamp),
        "max_range": float(max_range),
        "classes": list(classes),
        "num_points": int(points.shape[0]),
        "hero": {
            "id": int(hero_id),
            "type_id": hero_type_id,
        },
        "points": np.asarray(points, dtype=np.float32),
        "with_gt": bool(gt_payload is not None),
    }

    message.update(empty_gt_payload())
    if gt_payload is not None:
        message.update(dict(gt_payload))

    return message


def parse_lidar_message(message: Mapping[str, Any]) -> Dict[str, Any]:
    frame_id = int(message.get("frame", -1))
    points = np.asarray(message["points"], dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Invalid points shape: {points.shape}")

    gt_boxes = np.asarray(message.get("gt_boxes", np.zeros((0, 7))), dtype=np.float32)
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 7:
        raise ValueError(f"Invalid gt_boxes shape: {gt_boxes.shape}")

    gt_names = [str(name) for name in message.get("gt_names", [])]

    return {
        "frame": frame_id,
        "timestamp": float(message.get("timestamp", -1.0)),
        "points": points,
        "gt_boxes": gt_boxes,
        "gt_names": gt_names,
        "with_gt": bool(message.get("with_gt", False)),
    }


def build_prediction_message(
    *,
    input_message: Mapping[str, Any],
    points: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_names: Sequence[str],
) -> Dict[str, Any]:
    parsed = parse_lidar_message(input_message)
    return {
        "frame": parsed["frame"],
        "timestamp": parsed["timestamp"],
        "with_gt": parsed["with_gt"],
        "points": np.asarray(points, dtype=np.float32),
        "gt_boxes": parsed["gt_boxes"],
        "gt_names": parsed["gt_names"],
        "pred_boxes": np.asarray(pred_boxes, dtype=np.float32),
        "pred_scores": np.asarray(pred_scores, dtype=np.float32),
        "pred_names": list(pred_names),
    }
