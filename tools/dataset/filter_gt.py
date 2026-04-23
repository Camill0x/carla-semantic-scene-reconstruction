#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def count_points_in_box7(points_xyz: np.ndarray, box7: np.ndarray) -> int:
    x, y, z, dx, dy, dz, yaw = map(float, box7)

    rel = points_xyz - np.array([x, y, z], dtype=np.float64).reshape(1, 3)

    c = np.cos(-yaw)
    s = np.sin(-yaw)
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    local = rel @ rot.T

    inside = (np.abs(local[:, 0]) <= dx / 2.0) & (np.abs(local[:, 1]) <= dy / 2.0) & (np.abs(local[:, 2]) <= dz / 2.0)
    return int(np.count_nonzero(inside))


def load_frame(frame_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, Optional[dict], Optional[Path]]:
    points = np.load(frame_dir / "points.npy")
    gt_boxes = np.load(frame_dir / "gt_boxes.npy")
    gt_names = np.load(frame_dir / "gt_names.npy")

    with open(frame_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    objects_payload = None
    objects_path = None
    candidate_path = frame_dir / "objects.json"
    if candidate_path.exists():
        with open(candidate_path, "r", encoding="utf-8") as f:
            objects_payload = json.load(f)
        objects_path = candidate_path

    return points, gt_boxes, gt_names, meta, objects_payload, objects_path


def save_frame(
    frame_dir: Path,
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    meta: dict,
    objects_payload: Optional[dict] = None,
    objects_path: Optional[Path] = None,
) -> None:
    np.save(frame_dir / "gt_boxes.npy", gt_boxes)
    np.save(frame_dir / "gt_names.npy", gt_names)

    with open(frame_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if objects_payload is not None and objects_path is not None:
        with open(objects_path, "w", encoding="utf-8") as f:
            json.dump(objects_payload, f, indent=2)


def build_class_counts(gt_names: np.ndarray) -> Dict[str, int]:
    class_counts: Dict[str, int] = {}
    for name in gt_names.tolist():
        class_counts[name] = class_counts.get(name, 0) + 1
    return class_counts


def extract_objects(payload: Any) -> List[dict]:
    if isinstance(payload, dict):
        objects = payload.get("objects", [])
        return objects if isinstance(objects, list) else []
    return []


def normalize_meta_summary(meta: dict) -> dict:
    summary_keys = ("num_points", "classes", "num_objects", "class_counts", "num_lanes")
    summary = {}
    for key in summary_keys:
        if key in meta:
            summary[key] = meta.pop(key)

    return {
        **meta,
        **summary,
    }


def filter_single_frame(
    frame_dir: Path,
    min_gt_points: int,
    drop_empty_frames: bool,
    dry_run: bool,
) -> Tuple[bool, int, int, bool]:
    points, gt_boxes, gt_names, meta, objects_payload, objects_path = load_frame(frame_dir)

    if gt_boxes.shape[0] != gt_names.shape[0]:
        raise RuntimeError(
            f"Inconsistent GT shapes in {frame_dir}: " f"gt_boxes={gt_boxes.shape}, gt_names={gt_names.shape}"
        )

    points_xyz = points[:, :3]
    kept_indices: List[int] = []
    num_lidar_points_per_gt: List[int] = []

    for i, box in enumerate(gt_boxes):
        num_points = count_points_in_box7(points_xyz, box)
        num_lidar_points_per_gt.append(num_points)

        if num_points >= min_gt_points:
            kept_indices.append(i)

    original_count = int(gt_boxes.shape[0])
    kept_count = len(kept_indices)
    removed_count = original_count - kept_count

    if kept_count > 0:
        kept_boxes = gt_boxes[kept_indices]
        kept_names = gt_names[kept_indices]
    else:
        kept_boxes = np.zeros((0, 7), dtype=np.float32)
        kept_names = np.array([], dtype=gt_names.dtype if gt_names.size > 0 else "<U16")

    old_objects: List[dict] = []
    if isinstance(objects_payload, dict):
        old_objects = extract_objects(objects_payload)
    elif isinstance(meta.get("objects"), list):
        old_objects = meta.get("objects", [])

    kept_objects = [old_objects[i] for i in kept_indices] if len(old_objects) == original_count else old_objects
    class_counts = build_class_counts(kept_names)
    meta["postprocess_min_gt_points"] = int(min_gt_points)
    meta["num_objects"] = int(kept_count)
    meta["class_counts"] = class_counts

    if len(old_objects) == original_count:
        for out_idx, src_idx in enumerate(kept_indices):
            if out_idx < len(kept_objects):
                kept_objects[out_idx]["num_lidar_points"] = int(num_lidar_points_per_gt[src_idx])

    if isinstance(objects_payload, dict):
        objects_payload["objects"] = kept_objects
    else:
        meta["objects"] = kept_objects
        meta["gt_names"] = kept_names.tolist()

    meta = normalize_meta_summary(meta)

    should_delete_frame = drop_empty_frames and kept_count == 0

    if not dry_run:
        if should_delete_frame:
            for child in frame_dir.iterdir():
                child.unlink()
            frame_dir.rmdir()
        else:
            save_frame(
                frame_dir,
                kept_boxes,
                kept_names,
                meta,
                objects_payload=objects_payload,
                objects_path=objects_path,
            )

    return True, removed_count, kept_count, should_delete_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter GT boxes by number of LiDAR points")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset root containing frame_* folders")
    parser.add_argument(
        "--min-gt-points", type=int, required=True, help="Keep GT only if bbox contains at least N points"
    )
    parser.add_argument(
        "--drop-empty-frames",
        action="store_true",
        help="Delete frame directories that end up with zero GT objects",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be changed, do not modify files",
    )
    args = parser.parse_args()

    if args.min_gt_points < 0:
        raise ValueError("--min-gt-points must be >= 0")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not dataset_dir.is_dir():
        raise NotADirectoryError(dataset_dir)

    frame_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("frame_"))

    if not frame_dirs:
        print("[info] no frame_* directories found")
        return

    total_frames = 0
    total_removed_gt = 0
    total_kept_gt = 0
    total_deleted_frames = 0

    for frame_dir in frame_dirs:
        changed, removed_count, kept_count, deleted_frame = filter_single_frame(
            frame_dir=frame_dir,
            min_gt_points=args.min_gt_points,
            drop_empty_frames=args.drop_empty_frames,
            dry_run=args.dry_run,
        )

        total_frames += 1
        total_removed_gt += removed_count
        total_kept_gt += kept_count
        total_deleted_frames += int(deleted_frame)

        status = "DELETE_FRAME" if deleted_frame else "OK"
        prefix = "[dry-run]" if args.dry_run else "[done]"
        print(f"{prefix} {frame_dir.name} | kept={kept_count} | removed={removed_count} | status={status}")

    print(
        f"[summary] frames={total_frames} | kept_gt={total_kept_gt} | "
        f"removed_gt={total_removed_gt} | deleted_frames={total_deleted_frames}"
    )


if __name__ == "__main__":
    main()
