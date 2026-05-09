from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np


@dataclass(frozen=True)
class Objects3DPrediction:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    names: List[str]

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @classmethod
    def empty(cls) -> "Objects3DPrediction":
        return cls(
            boxes=np.zeros((0, 7), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            names=[],
        )

    @classmethod
    def from_detector_output(cls, pred_dict) -> "Objects3DPrediction":
        # Drop optional fields such as velocity; downstream code uses 7D boxes.
        return cls(
            boxes=pred_dict["pred_boxes"].detach().cpu().numpy()[:, :7].astype(np.float32),
            scores=pred_dict["pred_scores"].detach().cpu().numpy().astype(np.float32),
            labels=pred_dict["pred_labels"].detach().cpu().numpy().astype(np.int64),
            names=[],
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Objects3DPrediction":
        boxes = np.asarray(payload.get("boxes", np.zeros((0, 7))), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] < 7:
            boxes = np.zeros((0, 7), dtype=np.float32)
        boxes = boxes[:, :7]

        scores = np.asarray(payload.get("scores", np.zeros((boxes.shape[0],))), dtype=np.float32)
        labels = np.asarray(payload.get("labels", np.zeros((boxes.shape[0],))), dtype=np.int64)
        names = [str(name) for name in payload.get("names", [])]

        return cls(boxes=boxes, scores=scores, labels=labels, names=names)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "boxes": self.boxes,
            "scores": self.scores,
            "labels": self.labels,
            "names": self.names,
        }
