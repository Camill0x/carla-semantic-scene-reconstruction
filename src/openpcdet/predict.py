from dataclasses import dataclass
import time
from typing import Iterable, List, Sequence

import numpy as np
import torch
from pcdet.models import load_data_to_gpu

from src.openpcdet.runtime_dataset import RealtimeDataset


@dataclass(frozen=True)
class Objects3DPrediction:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    names: List[str]


def append_zero_timestamps(points4: np.ndarray) -> np.ndarray:
    timestamps = np.zeros((points4.shape[0], 1), dtype=np.float32)
    return np.hstack([points4.astype(np.float32), timestamps])


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def filter_object_predictions(
    pred_dict,
    class_names: Sequence[str],
    allowed_classes: Iterable[str],
    score_thresh: float,
) -> Objects3DPrediction:
    pred_boxes = pred_dict["pred_boxes"].detach().cpu().numpy()
    pred_scores = pred_dict["pred_scores"].detach().cpu().numpy()
    pred_labels = pred_dict["pred_labels"].detach().cpu().numpy()

    allowed_class_set = set(allowed_classes)
    keep = []
    names = []

    for index, (label_idx, score) in enumerate(zip(pred_labels, pred_scores)):
        class_name = class_names[label_idx - 1]
        if class_name not in allowed_class_set:
            continue
        if float(score) < score_thresh:
            continue
        keep.append(index)
        names.append(class_name)

    if not keep:
        return Objects3DPrediction(
            boxes=np.zeros((0, 7), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            names=[],
        )

    keep_array = np.array(keep, dtype=np.int64)

    # Drop optional fields such as velocity; downstream code uses 7D boxes.
    return Objects3DPrediction(
        boxes=pred_boxes[keep_array, :7].astype(np.float32),
        scores=pred_scores[keep_array].astype(np.float32),
        labels=pred_labels[keep_array].astype(np.int64),
        names=names,
    )


def run_inference(
    dataset: RealtimeDataset,
    model,
    points4: np.ndarray,
    frame_id: int,
    *,
    return_forward_time: bool = False,
):
    points5 = append_zero_timestamps(points4)
    input_dict = {
        "points": points5,
        "frame_id": frame_id,
    }
    data_dict = dataset.prepare_data(data_dict=input_dict)
    batch_dict = dataset.collate_batch([data_dict])
    load_data_to_gpu(batch_dict)

    if return_forward_time:
        synchronize_cuda()
        t0 = time.perf_counter()

    with torch.no_grad():
        pred_dicts, _ = model.forward(batch_dict)

    if return_forward_time:
        synchronize_cuda()
        forward_time_s = time.perf_counter() - t0
        return pred_dicts[0], forward_time_s

    return pred_dicts[0]
