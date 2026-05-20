from typing import Iterable, Sequence

import numpy as np

from src.openpcdet.prediction import Objects3DPrediction


def filter_object_predictions(
    objects_3d: Objects3DPrediction,
    class_names: Sequence[str],
    allowed_classes: Iterable[str],
    score_thresh: float,
) -> Objects3DPrediction:
    """Keep only the predicted objects whose class names are allowed."""
    allowed_class_set = set(allowed_classes)
    keep = []
    names = []

    for index, (label_idx, score) in enumerate(zip(objects_3d.labels, objects_3d.scores)):
        class_name = class_names[label_idx - 1]
        if class_name not in allowed_class_set:
            continue
        if float(score) < score_thresh:
            continue
        keep.append(index)
        names.append(class_name)

    if not keep:
        return Objects3DPrediction.empty()

    keep_array = np.array(keep, dtype=np.int64)

    return Objects3DPrediction(
        boxes=objects_3d.boxes[keep_array],
        scores=objects_3d.scores[keep_array],
        labels=objects_3d.labels[keep_array],
        names=names,
    )
