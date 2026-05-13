import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.common.dataset import DatasetSplits
from src.common.typing_aliases import JsonDict


def frame_has_class_counts(meta: JsonDict, class_names: Sequence[str]) -> bool:
    """Return whether frame metadata contains counts for all requested classes."""
    class_counts = meta.get("class_counts", {})
    if not isinstance(class_counts, dict):
        return False
    return any(class_name in class_counts and int(class_counts[class_name]) > 0 for class_name in class_names)


def load_frame_info(frame_dir: Path, output_root: Path, class_names: Sequence[str]) -> Optional[JsonDict]:
    """Build one OpenPCDet sample-info record from a recorded CARLA frame."""
    points_path = frame_dir / "points.npy"
    gt_boxes_path = frame_dir / "gt_boxes.npy"
    gt_names_path = frame_dir / "gt_names.npy"
    meta_path = frame_dir / "meta.json"

    required_paths = [points_path, gt_boxes_path, gt_names_path, meta_path]
    if not all(path.exists() for path in required_paths):
        return None

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        return None

    if int(meta.get("num_objects", 0)) <= 0:
        return None
    if not frame_has_class_counts(meta, class_names):
        return None

    gt_boxes = np.load(gt_boxes_path).astype(np.float32)
    gt_names = np.load(gt_names_path)

    if gt_boxes.ndim != 2:
        raise ValueError(f"Invalid gt_boxes shape in {gt_boxes_path}: {gt_boxes.shape}")
    if gt_boxes.shape[0] != len(gt_names):
        raise ValueError(f"gt_boxes / gt_names mismatch in {frame_dir}")

    mask = np.array([str(name) in class_names for name in gt_names], dtype=bool)
    gt_boxes = gt_boxes[mask]
    gt_names = gt_names[mask].astype(str)

    if gt_boxes.shape[0] == 0:
        return None

    run_id = frame_dir.parent.name
    frame_name = frame_dir.name
    sample_id = f"{run_id}__{frame_name}"

    return {
        "frame_id": sample_id,
        "point_cloud": {
            "lidar_path": os.path.relpath(points_path, output_root),
            "num_features": 4,
        },
        "metadata": {
            "frame": int(meta["frame"]) if "frame" in meta else int(meta.get("sim_frame", -1)),
            "timestamp": float(meta.get("timestamp", -1.0)),
            "map": meta.get("map", ""),
            "source_run": run_id,
            "source_frame_dir": frame_name,
        },
        "annos": {
            "name": np.array(gt_names),
            "gt_boxes_lidar": gt_boxes,
        },
    }


def load_infos(frame_dirs: Sequence[Path], output_root: Path, class_names: Sequence[str]) -> List[JsonDict]:
    """Load OpenPCDet sample-info records for a list of recorded frames."""
    infos: List[JsonDict] = []
    for frame_dir in frame_dirs:
        info = load_frame_info(frame_dir, output_root, class_names)
        if info is not None:
            infos.append(info)

    frame_ids = [info["frame_id"] for info in infos]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Duplicate sample ids detected after dataset preparation")

    return infos


def write_split(sample_ids: Sequence[str], output_path: Path) -> None:
    """Write a sample-id split file for OpenPCDet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            handle.write(f"{sample_id}\n")


def write_infos(output_root: Path, splits: DatasetSplits[JsonDict]) -> Tuple[Path, Path, Path]:
    """Write OpenPCDet split files and info pickles for the prepared dataset."""
    infos_root = output_root / "infos"
    image_sets_root = output_root / "ImageSets"
    infos_root.mkdir(parents=True, exist_ok=True)
    image_sets_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": infos_root / "infos_train.pkl",
        "val": infos_root / "infos_val.pkl",
        "test": infos_root / "infos_test.pkl",
    }
    split_infos = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }

    for split_name, infos in split_infos.items():
        with paths[split_name].open("wb") as handle:
            pickle.dump(infos, handle)
        write_split(
            [info["frame_id"] for info in infos],
            image_sets_root / f"{split_name}.txt",
        )

    return paths["train"], paths["val"], paths["test"]
