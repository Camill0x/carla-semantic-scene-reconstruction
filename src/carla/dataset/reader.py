import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


def load_frame(frame_dir: Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    meta_path = frame_dir / "meta.json"
    points_path = frame_dir / "points.npy"
    ego_box_path = frame_dir / "ego_box.npy"
    objects_path = frame_dir / "objects.json"
    lanes_path = frame_dir / "lanes.json"

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not points_path.exists():
        raise FileNotFoundError(points_path)
    if not ego_box_path.exists():
        raise FileNotFoundError(ego_box_path)
    if not objects_path.exists():
        raise FileNotFoundError(objects_path)
    if not lanes_path.exists():
        raise FileNotFoundError(lanes_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    points = np.load(points_path)
    ego_box = np.load(ego_box_path)

    with objects_path.open("r", encoding="utf-8") as handle:
        objects = json.load(handle).get("objects", [])

    with lanes_path.open("r", encoding="utf-8") as handle:
        lanes = json.load(handle).get("lanes", [])

    return meta, points, ego_box, objects, lanes
