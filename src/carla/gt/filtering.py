from typing import List, Sequence, Tuple

import numpy as np

from src.common.typing_aliases import Float32Array, Float64Array, ObjectDict, StrArray


def count_points_in_box7(points_xyz: Float64Array, box7: Float32Array) -> int:
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


def filter_gt(
    points: Float32Array,
    objects: Sequence[ObjectDict],
    gt_boxes: Float32Array,
    gt_names: StrArray,
    min_points_in_box: int,
) -> Tuple[List[ObjectDict], Float32Array, StrArray]:
    if min_points_in_box <= 0 or gt_boxes.shape[0] == 0:
        return list(objects), gt_boxes, gt_names

    if gt_boxes.shape[0] != gt_names.shape[0]:
        raise ValueError(f"gt_boxes / gt_names mismatch: {gt_boxes.shape} vs {gt_names.shape}")
    if len(objects) != gt_boxes.shape[0]:
        raise ValueError(f"objects / gt_boxes mismatch: {len(objects)} vs {gt_boxes.shape[0]}")

    points_xyz = np.asarray(points[:, :3], dtype=np.float64)
    keep_indices = [
        index for index, box in enumerate(gt_boxes) if count_points_in_box7(points_xyz, box) >= min_points_in_box
    ]

    if len(keep_indices) == gt_boxes.shape[0]:
        return list(objects), gt_boxes, gt_names

    kept_objects = [dict(objects[index]) for index in keep_indices]
    if keep_indices:
        kept_boxes = gt_boxes[keep_indices]
        kept_names = gt_names[keep_indices]
    else:
        kept_boxes = np.zeros((0, 7), dtype=gt_boxes.dtype)
        kept_names = np.array([], dtype=gt_names.dtype if gt_names.size > 0 else "<U16")

    return kept_objects, kept_boxes, kept_names
