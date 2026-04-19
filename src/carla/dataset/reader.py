import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_frame(frame_dir: str) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    points_path = os.path.join(frame_dir, "points.npy")
    meta_path = os.path.join(frame_dir, "meta.json")
    ego_box_path = os.path.join(frame_dir, "ego_box.npy")

    if not os.path.exists(points_path):
        raise FileNotFoundError(points_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    points = np.load(points_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ego_box = None
    if os.path.exists(ego_box_path):
        ego_box = np.load(ego_box_path)
    elif "ego_vehicle_box" in meta:
        ego_box = np.asarray(meta["ego_vehicle_box"], dtype=np.float32)

    return points, meta, ego_box
