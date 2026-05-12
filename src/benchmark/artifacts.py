import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def create_benchmark_output_dir(*, run_dir: Path, model_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("results") / "benchmark" / run_dir.name / model_name
    output_dir = base_dir / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def build_openpcdet_meta(
    *,
    run_dir: Path,
    cfg_file: Path,
    ckpt: Path,
    frames_total: int,
    warmup: int,
    score_thresh: float,
    point_stride: int,
    save_predictions: bool,
    model_class_names: Sequence[str],
    dataset_meta: Mapping[str, Any],
    created_at: str,
) -> Mapping[str, Any]:
    return {
        "model": "openpcdet",
        "created_at": created_at,
        "run_name": run_dir.name,
        "run_dir": run_dir.expanduser().resolve(),
        "cfg_file": cfg_file.expanduser().resolve(),
        "ckpt": ckpt.expanduser().resolve(),
        "frames_total": frames_total,
        "warmup": warmup,
        "score_thresh": score_thresh,
        "point_stride": point_stride,
        "save_predictions": save_predictions,
        "model_class_names": list(model_class_names),
        "dataset_gt_classes": dataset_meta.get("classes"),
        "dataset": {
            "map": dataset_meta.get("map"),
            "lidar": dataset_meta.get("lidar"),
        },
    }


def build_lanedet_meta(
    *,
    run_dir: Path,
    config: Path,
    ckpt: Path,
    frames_total: int,
    warmup: int,
    score_thresh: float,
    save_predictions: bool,
    dataset_meta: Mapping[str, Any],
    created_at: str,
) -> Mapping[str, Any]:
    return {
        "model": "lanedet",
        "created_at": created_at,
        "run_name": run_dir.name,
        "run_dir": run_dir.expanduser().resolve(),
        "config": config.expanduser().resolve(),
        "ckpt": ckpt.expanduser().resolve(),
        "frames_total": frames_total,
        "warmup": warmup,
        "score_thresh": score_thresh,
        "save_predictions": save_predictions,
        "dataset": {
            "map": dataset_meta.get("map"),
            "front_camera": dataset_meta.get("front_camera"),
        },
    }


def write_meta_json(path: Path, *, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonify(dict(payload)), handle, indent=2)
