import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


@dataclass(frozen=True)
class DatasetFrame:
    frame_dir: Path
    meta: Dict[str, Any]
    points: np.ndarray
    ego_box: np.ndarray
    gt_boxes: np.ndarray
    gt_names: List[str]
    objects: List[Dict[str, Any]]
    lanes: List[Dict[str, Any]]
    image_rgb: np.ndarray


@dataclass(frozen=True)
class PointsFrame:
    frame_dir: Path
    meta: Dict[str, Any]
    points: np.ndarray


@dataclass(frozen=True)
class CameraFrame:
    frame_dir: Path
    meta: Dict[str, Any]
    image_rgb: np.ndarray


def _require_path(frame_dir: Path, filename: str) -> Path:
    path = frame_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def _load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_points_frame(frame_dir: Path) -> PointsFrame:
    meta = _load_json(_require_path(frame_dir, "meta.json"))
    points = _load_npy(_require_path(frame_dir, "points.npy"))

    return PointsFrame(
        frame_dir=frame_dir,
        meta=meta,
        points=points,
    )


def load_camera_frame(frame_dir: Path) -> CameraFrame:
    meta = _load_json(_require_path(frame_dir, "meta.json"))
    image_rgb = _load_image_rgb(_require_path(frame_dir, "front_rgb.png"))
    return CameraFrame(
        frame_dir=frame_dir,
        meta=meta,
        image_rgb=image_rgb,
    )


def load_dataset_frame(frame_dir: Path) -> DatasetFrame:
    meta = _load_json(_require_path(frame_dir, "meta.json"))
    points = _load_npy(_require_path(frame_dir, "points.npy"))
    ego_box = _load_npy(_require_path(frame_dir, "ego_box.npy"))
    gt_boxes = _load_npy(_require_path(frame_dir, "gt_boxes.npy"))
    gt_names = _load_npy(_require_path(frame_dir, "gt_names.npy"))
    objects_payload = _load_json(_require_path(frame_dir, "objects.json"))
    lanes_payload = _load_json(_require_path(frame_dir, "lanes.json"))
    image_rgb = _load_image_rgb(_require_path(frame_dir, "front_rgb.png"))

    return DatasetFrame(
        frame_dir=frame_dir,
        meta=meta,
        points=points,
        ego_box=ego_box,
        gt_boxes=gt_boxes,
        gt_names=[str(name) for name in gt_names.tolist()],
        objects=objects_payload.get("objects", []),
        lanes=lanes_payload.get("lanes", []),
        image_rgb=image_rgb,
    )


def iter_frame_dirs(run_dir: Path) -> List[Path]:
    return sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("frame_"))
