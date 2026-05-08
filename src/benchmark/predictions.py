import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def save_objects_prediction(path: Path, *, boxes: np.ndarray, scores: np.ndarray, names) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        boxes=np.asarray(boxes, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        names=np.asarray(list(names), dtype=str),
    )


def load_objects_prediction(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "pred_boxes": np.asarray(data["boxes"], dtype=np.float32),
            "pred_scores": np.asarray(data["scores"], dtype=np.float32),
            "pred_names": [str(name) for name in data["names"].tolist()],
        }


def _jsonify(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_lanes_prediction(path: Path, lanes_3d: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonify(dict(lanes_3d)), handle, indent=2)


def load_lanes_prediction(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    strips = [np.asarray(strip, dtype=np.float32) for strip in payload.get("strips", [])]
    return {
        **payload,
        "strips": strips,
        "scores": [float(score) for score in payload.get("scores", [])],
        "names": [str(name) for name in payload.get("names", [])],
    }
