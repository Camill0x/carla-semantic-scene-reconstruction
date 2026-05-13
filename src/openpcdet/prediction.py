from dataclasses import dataclass
from typing import Any, List, Mapping

import numpy as np

from src.common.typing_aliases import Float32Array, IntArray, JsonDict


@dataclass(frozen=True)
class Objects3DPrediction:
    boxes: Float32Array
    scores: Float32Array
    labels: IntArray
    names: List[str]

    def __len__(self) -> int:
        """Return the number of predicted 3D objects."""
        return int(self.boxes.shape[0])

    @classmethod
    def empty(cls) -> "Objects3DPrediction":
        """Return an empty 3D object prediction container."""
        return cls(
            boxes=np.zeros((0, 7), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            names=[],
        )

    @classmethod
    def from_detector_output(cls, pred_dict: Mapping[str, Any]) -> "Objects3DPrediction":
        # Drop optional fields such as velocity; downstream code uses 7D boxes.
        """Build a prediction container from an OpenPCDet detector output."""
        return cls(
            boxes=pred_dict["pred_boxes"].detach().cpu().numpy()[:, :7].astype(np.float32),
            scores=pred_dict["pred_scores"].detach().cpu().numpy().astype(np.float32),
            labels=pred_dict["pred_labels"].detach().cpu().numpy().astype(np.int64),
            names=[],
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Objects3DPrediction":
        """Build a prediction container from a serialized payload."""
        boxes = np.asarray(payload.get("boxes", np.zeros((0, 7))), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] < 7:
            boxes = np.zeros((0, 7), dtype=np.float32)
        boxes = boxes[:, :7]

        scores = np.asarray(payload.get("scores", np.zeros((boxes.shape[0],))), dtype=np.float32)
        labels = np.asarray(payload.get("labels", np.zeros((boxes.shape[0],))), dtype=np.int64)
        names = [str(name) for name in payload.get("names", [])]

        return cls(boxes=boxes, scores=scores, labels=labels, names=names)

    def to_payload(self) -> JsonDict:
        """Serialize the prediction container into a transport payload."""
        return {
            "boxes": self.boxes,
            "scores": self.scores,
            "labels": self.labels,
            "names": self.names,
        }
