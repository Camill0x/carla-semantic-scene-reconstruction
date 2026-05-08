import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping

import numpy as np


def summarize_frame_metrics(frame_metrics: List[Mapping[str, float]], *, model: str, warmup: int) -> Dict[str, Any]:
    measured = [dict(item) for item in frame_metrics if not bool(item.get("warmup", 0.0))]
    summary: Dict[str, Any] = {
        "model": model,
        "frames_total": len(frame_metrics),
        "frames_measured": len(measured),
        "warmup": warmup,
    }
    if not measured:
        return summary

    keys = [key for key in measured[0] if key.endswith("_ms")]
    for key in keys:
        values = np.asarray([float(item[key]) for item in measured], dtype=np.float64)
        summary[f"mean_{key}"] = float(mean(values))

    infer_values = np.asarray([float(item["infer_ms"]) for item in measured], dtype=np.float64)
    summary["inference_fps"] = float(1000.0 / max(float(mean(infer_values)), 1e-9))
    if "num_points" in measured[0]:
        summary["mean_num_points_per_frame"] = int(round(mean(float(item["num_points"]) for item in measured)))
    return summary


def round_metrics(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {key: round_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_metrics(item) for item in value]
    return value


def write_metrics_json(path: Path, *, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(round_metrics(dict(summary)), handle, indent=2)
