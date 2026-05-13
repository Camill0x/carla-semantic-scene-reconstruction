import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, cast

import cv2
import numpy as np

from src.common.typing_aliases import ArrayAny, ImageArray, JsonDict


@dataclass(frozen=True)
class DatasetFrame:
    frame_dir: Path
    meta: JsonDict
    points: ArrayAny
    ego_box: ArrayAny
    gt_boxes: ArrayAny
    gt_names: List[str]
    objects: List[JsonDict]
    lanes: List[JsonDict]
    image_rgb: ImageArray


@dataclass(frozen=True)
class PointsFrame:
    frame_dir: Path
    meta: JsonDict
    points: ArrayAny


@dataclass(frozen=True)
class CameraFrame:
    frame_dir: Path
    meta: JsonDict
    image_rgb: ImageArray


def _require_path(frame_dir: Path, filename: str) -> Path:
    """Return a required frame file path or raise if it is missing."""
    path = frame_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_json(path: Path) -> JsonDict:
    """Load and validate a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def _load_npy(path: Path) -> ArrayAny:
    """Load a NumPy array from disk."""
    return np.asarray(np.load(path))


def _load_image_rgb(path: Path) -> ImageArray:
    """Load an image from disk and convert it from BGR to RGB."""
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return np.asarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def load_points_frame(frame_dir: Path) -> PointsFrame:
    """Load the LiDAR-only view of a recorded dataset frame."""
    meta = _load_json(_require_path(frame_dir, "meta.json"))
    points = _load_npy(_require_path(frame_dir, "points.npy"))

    return PointsFrame(
        frame_dir=frame_dir,
        meta=meta,
        points=points,
    )


def load_camera_frame(frame_dir: Path) -> CameraFrame:
    """Load the camera-only view of a recorded dataset frame."""
    meta = _load_json(_require_path(frame_dir, "meta.json"))
    image_rgb = _load_image_rgb(_require_path(frame_dir, "front_rgb.png"))
    return CameraFrame(
        frame_dir=frame_dir,
        meta=meta,
        image_rgb=image_rgb,
    )


def load_dataset_frame(frame_dir: Path) -> DatasetFrame:
    """Load all multimodal payloads for a recorded dataset frame."""
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
        objects=[cast(JsonDict, item) for item in objects_payload.get("objects", []) if isinstance(item, dict)],
        lanes=[cast(JsonDict, item) for item in lanes_payload.get("lanes", []) if isinstance(item, dict)],
        image_rgb=image_rgb,
    )


def iter_frame_dirs(run_dir: Path) -> List[Path]:
    """Return the sorted frame directories for the provided runs."""
    return sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("frame_"))
