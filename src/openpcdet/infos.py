import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.dataset import DatasetSplits
from src.common.typing_aliases import BoolArray, Float32Array, JsonDict, StrArray


def frame_has_class_counts(meta: JsonDict, class_names: Sequence[str]) -> bool:
    """Return whether frame metadata contains counts for all requested classes."""
    class_counts = meta.get("class_counts", {})
    if not isinstance(class_counts, dict):
        return False
    for class_name in class_names:
        value = class_counts.get(class_name)
        if isinstance(value, (int, float, str)) and int(value) > 0:
            return True
    return False


def validate_filtered_annos(gt_boxes: Float32Array, gt_names: StrArray, frame_dir: Path) -> bool:
    """Validate the filtered annotations kept for one frame."""
    if gt_boxes.ndim != 2:
        raise ValueError(f"Invalid gt_boxes shape in {frame_dir / 'gt_boxes.npy'}: {gt_boxes.shape}")
    if gt_boxes.shape[0] != len(gt_names):
        raise ValueError(f"gt_boxes / gt_names mismatch in {frame_dir}")
    return gt_boxes.shape[0] > 0


def count_filtered_objects_by_class(gt_names: StrArray, class_names: Sequence[str]) -> Dict[str, int]:
    """Count kept objects per class after filtering to the requested class set."""
    counts: Dict[str, int] = {class_name: 0 for class_name in class_names}
    for name in gt_names.tolist():
        class_name = str(name)
        if class_name in counts:
            counts[class_name] += 1
    return counts


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

    gt_boxes: Float32Array = np.load(gt_boxes_path).astype(np.float32)
    gt_names: StrArray = np.load(gt_names_path).astype(str)

    mask: BoolArray = np.array([str(name) in class_names for name in gt_names], dtype=bool)
    gt_boxes = gt_boxes[mask]
    gt_names = gt_names[mask]

    if not validate_filtered_annos(gt_boxes, gt_names, frame_dir):
        return None

    run_id = frame_dir.parent.name
    frame_name = frame_dir.name
    sample_id = f"{run_id}__{frame_name}"

    return {
        "frame_id": sample_id,
        "point_cloud": {
            "lidar_path": os.path.relpath(points_path, output_root),
        },
        "annos": {
            "name": np.array(gt_names),
            "gt_boxes_lidar": gt_boxes,
        },
    }


def load_infos(
    frame_dirs: Sequence[Path], output_root: Path, class_names: Sequence[str]
) -> Tuple[List[JsonDict], JsonDict]:
    """Load OpenPCDet sample-info records for a list of recorded frames with preparation stats."""
    infos: List[JsonDict] = []
    skipped_no_objects = 0
    skipped_other = 0
    total_objects = 0
    objects_per_class_total: Dict[str, int] = {class_name: 0 for class_name in class_names}
    objects_per_class_min: Dict[str, Optional[int]] = {class_name: None for class_name in class_names}
    objects_per_class_max: Dict[str, int] = {class_name: 0 for class_name in class_names}

    for frame_dir in frame_dirs:
        points_path = frame_dir / "points.npy"
        gt_boxes_path = frame_dir / "gt_boxes.npy"
        gt_names_path = frame_dir / "gt_names.npy"
        meta_path = frame_dir / "meta.json"
        required_paths = [points_path, gt_boxes_path, gt_names_path, meta_path]

        if not all(path.exists() for path in required_paths):
            skipped_other += 1
            continue

        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if not isinstance(meta, dict):
            skipped_other += 1
            continue

        if int(meta.get("num_objects", 0)) <= 0:
            skipped_no_objects += 1
            continue
        if not frame_has_class_counts(meta, class_names):
            skipped_other += 1
            continue

        info = load_frame_info(frame_dir, output_root, class_names)
        if info is None:
            skipped_other += 1
            continue

        infos.append(info)
        gt_names = info["annos"]["name"]
        class_counts = count_filtered_objects_by_class(gt_names, class_names)
        frame_total = sum(class_counts.values())
        total_objects += frame_total
        for class_name in class_names:
            count = int(class_counts[class_name])
            objects_per_class_total[class_name] += count
            current_min = objects_per_class_min[class_name]
            objects_per_class_min[class_name] = count if current_min is None else min(current_min, count)
            objects_per_class_max[class_name] = max(objects_per_class_max[class_name], count)

    frame_ids = [info["frame_id"] for info in infos]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Duplicate sample ids detected after dataset preparation")

    usable_samples = len(infos)
    objects_per_class: JsonDict = {}
    for class_name in class_names:
        total = int(objects_per_class_total[class_name])
        min_count = objects_per_class_min[class_name]
        min_value = 0 if min_count is None else int(min_count)
        max_value = int(objects_per_class_max[class_name])
        avg_value = (float(total) / float(usable_samples)) if usable_samples > 0 else 0.0
        objects_per_class[class_name] = {
            "total": total,
            "avg": avg_value,
            "min": min_value,
            "max": max_value,
        }

    stats: JsonDict = {
        "total_frames": len(frame_dirs),
        "usable_samples": usable_samples,
        "skipped_no_objects": skipped_no_objects,
        "skipped_other": skipped_other,
        "total_objects": total_objects,
        "objects_per_class": objects_per_class,
    }
    return infos, stats


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
