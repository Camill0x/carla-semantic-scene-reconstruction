from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np

from src.common.typing_aliases import Float32Array, JsonDict


@dataclass(frozen=True)
class Lanes2DPrediction:
    strips: List[Float32Array]
    scores: Float32Array
    names: List[str]

    def __len__(self) -> int:
        """Return the number of predicted 2D lanes."""
        return len(self.strips)

    @classmethod
    def empty(cls) -> "Lanes2DPrediction":
        """Return an empty 2D lane prediction container."""
        return cls(strips=[], scores=np.zeros((0,), dtype=np.float32), names=[])

    @classmethod
    def from_detector_output(cls, lanes_2d: Sequence[Tuple[Float32Array, float]]) -> "Lanes2DPrediction":
        """Handle from detector output."""
        strips: List[Float32Array] = []
        scores: List[float] = []
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
        """Build a 2D lane prediction container from a serialized payload."""
        strips: List[Float32Array] = []
        for strip in payload.get("strips", []):
            points = np.asarray(strip, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 2:
                strips.append(points)
        return cls(
            strips=strips,
            scores=np.asarray(payload.get("scores", []), dtype=np.float32),
            names=[str(name) for name in payload.get("names", [])],
        )

    def to_payload(self) -> JsonDict:
        """Serialize the 2D lane prediction container into a transport payload."""
        return {
            "strips": self.strips,
            "scores": self.scores,
            "names": self.names,
        }


@dataclass(frozen=True)
class Lanes3DPrediction:
    strips: List[Float32Array]
    scores: Float32Array
    names: List[str]

    def __len__(self) -> int:
        """Return the number of predicted 3D lanes."""
        return len(self.strips)

    @classmethod
    def empty(cls) -> "Lanes3DPrediction":
        """Return an empty 3D lane prediction container."""
        return cls(strips=[], scores=np.zeros((0,), dtype=np.float32), names=[])

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Lanes3DPrediction":
        """Build a 3D lane prediction container from a serialized payload."""
        strips: List[Float32Array] = []
        for strip in payload.get("strips", []):
            points = np.asarray(strip, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 3:
                strips.append(points)
        return cls(
            strips=strips,
            scores=np.asarray(payload.get("scores", []), dtype=np.float32),
            names=[str(name) for name in payload.get("names", [])],
        )

    def to_payload(self) -> JsonDict:
        """Serialize the 3D lane prediction container into a transport payload."""
        return {
            "strips": self.strips,
            "scores": self.scores,
            "names": self.names,
        }
