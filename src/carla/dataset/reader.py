import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_frame(frame_dir: Path) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    points_path = frame_dir / "points.npy"
    meta_path = frame_dir / "meta.json"
    ego_box_path = frame_dir / "ego_box.npy"

    if not points_path.exists():
        raise FileNotFoundError(points_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    points = np.load(points_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ego_box = None
    if ego_box_path.exists():
        ego_box = np.load(ego_box_path)
    elif "ego_vehicle_box" in meta:
        ego_box = np.asarray(meta["ego_vehicle_box"], dtype=np.float32)

    return points, meta, ego_box
