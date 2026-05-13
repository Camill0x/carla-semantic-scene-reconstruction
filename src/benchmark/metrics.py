import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from src.common.typing_aliases import JsonDict


def summarize_frame_metrics(frame_metrics: Sequence[Mapping[str, float]], *, model: str, warmup: int) -> JsonDict:
    """Summarize per-frame benchmark timings into aggregate metrics."""
    measured = [dict(item) for item in frame_metrics if not bool(item.get("warmup", 0.0))]
    summary: JsonDict = {
        "model": model,
        "frames_total": len(frame_metrics),
        "frames_measured": len(measured),
        "warmup": warmup,
    }
    if not measured:
        return summary

    if "model_forward_ms" in measured[0]:
        model_values = np.asarray([float(item["model_forward_ms"]) for item in measured], dtype=np.float64)
        summary["mean_model_forward_ms"] = float(mean(model_values))
        summary["model_fps"] = float(1000.0 / max(float(mean(model_values)), 1e-9))
    if "runtime_ms" in measured[0]:
        runtime_values = np.asarray([float(item["runtime_ms"]) for item in measured], dtype=np.float64)
        summary["mean_runtime_ms"] = float(mean(runtime_values))
        summary["runtime_fps"] = float(1000.0 / max(float(mean(runtime_values)), 1e-9))
    if "num_points" in measured[0]:
        summary["mean_num_points_per_frame"] = int(round(mean(float(item["num_points"]) for item in measured)))
    return summary


def round_metrics(value: Any) -> Any:
    """Recursively round floating-point metrics for JSON output."""
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {key: round_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_metrics(item) for item in value]
    return value


def write_metrics_json(path: Path, *, summary: Mapping[str, Any]) -> None:
    """Write rounded benchmark metrics to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(round_metrics(dict(summary)), handle, indent=2)
