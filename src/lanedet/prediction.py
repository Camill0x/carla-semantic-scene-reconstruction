from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Lanes2DPrediction:
    strips: List[np.ndarray]
    scores: np.ndarray
    names: List[str]

    def __len__(self) -> int:
        return len(self.strips)

    @classmethod
    def empty(cls) -> "Lanes2DPrediction":
        return cls(strips=[], scores=np.zeros((0,), dtype=np.float32), names=[])

    @classmethod
    def from_detector_output(cls, lanes_2d: Sequence[Tuple[np.ndarray, float]]) -> "Lanes2DPrediction":
        strips = []
        scores = []
        names = []

        for index, (points, score) in enumerate(lanes_2d):
            points_array = np.asarray(points, dtype=np.float32)
            if points_array.ndim != 2 or points_array.shape[0] < 2 or points_array.shape[1] != 2:
                continue
            strips.append(points_array)
            scores.append(float(score))
            names.append(f"lane_{index}")

        return cls(strips=strips, scores=np.asarray(scores, dtype=np.float32), names=names)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Lanes2DPrediction":
        strips = []
        for strip in payload.get("strips", []):
            points = np.asarray(strip, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 2:
                strips.append(points)
        return cls(
            strips=strips,
            scores=np.asarray(payload.get("scores", []), dtype=np.float32),
            names=[str(name) for name in payload.get("names", [])],
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "strips": self.strips,
            "scores": self.scores,
            "names": self.names,
        }


@dataclass(frozen=True)
class Lanes3DPrediction:
    strips: List[np.ndarray]
    scores: np.ndarray
    names: List[str]

    def __len__(self) -> int:
        return len(self.strips)

    @classmethod
    def empty(cls) -> "Lanes3DPrediction":
        return cls(strips=[], scores=np.zeros((0,), dtype=np.float32), names=[])

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Lanes3DPrediction":
        strips = []
        for strip in payload.get("strips", []):
            points = np.asarray(strip, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 3:
                strips.append(points)
        return cls(
            strips=strips,
            scores=np.asarray(payload.get("scores", []), dtype=np.float32),
            names=[str(name) for name in payload.get("names", [])],
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "strips": self.strips,
            "scores": self.scores,
            "names": self.names,
        }
