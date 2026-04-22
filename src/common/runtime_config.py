import json
from pathlib import Path
from typing import Any, Dict, Mapping

from src.common.config import (
    CarlaConnectionConfig,
    CollectorConfig,
    LidarConfig,
    LiveInferenceConfig,
    LiveProducerConfig,
    LiveVisualizerConfig,
)

DEFAULT_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "runtime.json"


def _read_runtime_config() -> Dict[str, Any]:
    with DEFAULT_RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Runtime config at {DEFAULT_RUNTIME_CONFIG_PATH} must contain a JSON object")
    return data


def _get_section(root: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    current: Any = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            joined_path = ".".join(path)
            raise KeyError(f"Missing runtime config section: {joined_path}")
        current = current[key]
    if not isinstance(current, Mapping):
        joined_path = ".".join(path)
        raise ValueError(f"Runtime config section {joined_path} must be an object")
    return current


def _require_value(section: Mapping[str, Any], key: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing runtime config value: {key}")
    return section[key]


def load_carla_connection_config() -> CarlaConnectionConfig:
    data = _read_runtime_config()
    section = _get_section(data, "carla")
    return CarlaConnectionConfig(
        host=str(_require_value(section, "host")),
        port=int(_require_value(section, "port")),
    )


def load_lidar_config() -> LidarConfig:
    data = _read_runtime_config()
    section = _get_section(data, "lidar")
    return LidarConfig(
        max_range=float(_require_value(section, "max_range")),
        channels=int(_require_value(section, "channels")),
        points_per_second=int(_require_value(section, "points_per_second")),
        upper_fov=float(_require_value(section, "upper_fov")),
        lower_fov=float(_require_value(section, "lower_fov")),
        ego_bbox_padding=float(_require_value(section, "ego_bbox_padding")),
    )


def build_collector_config(*, output_dir: str, num_frames: int, every_nth: int) -> CollectorConfig:
    return CollectorConfig(
        carla=load_carla_connection_config(),
        lidar=load_lidar_config(),
        output_dir=output_dir,
        num_frames=num_frames,
        every_nth=every_nth,
    )


def build_live_producer_config(*, with_gt: bool, every_nth: int) -> LiveProducerConfig:
    data = _read_runtime_config()
    section = _get_section(data, "streaming", "producer")
    return LiveProducerConfig(
        carla=load_carla_connection_config(),
        lidar=load_lidar_config(),
        zmq_bind=str(_require_value(section, "zmq_bind")),
        every_nth=every_nth,
        with_gt=with_gt,
    )


def build_live_inference_config(
    *,
    cfg_file: str,
    ckpt: str,
    score_thresh: float,
    point_stride: int,
) -> LiveInferenceConfig:
    data = _read_runtime_config()
    section = _get_section(data, "streaming", "inference")
    return LiveInferenceConfig(
        cfg_file=cfg_file,
        ckpt=ckpt,
        zmq_in=str(_require_value(section, "zmq_in")),
        zmq_out=str(_require_value(section, "zmq_out")),
        score_thresh=score_thresh,
        point_stride=point_stride,
    )


def build_live_visualizer_config(
    *,
    show_grid: bool,
    hide_points: bool,
    hide_gt: bool,
) -> LiveVisualizerConfig:
    data = _read_runtime_config()
    section = _get_section(data, "streaming", "visualizer")
    return LiveVisualizerConfig(
        zmq_connect=str(_require_value(section, "zmq_connect")),
        show_grid=show_grid,
        hide_points=hide_points,
        hide_gt=hide_gt,
        point_radius=float(_require_value(section, "point_radius")),
        pred_line_radius=float(_require_value(section, "pred_line_radius")),
        gt_line_radius=float(_require_value(section, "gt_line_radius")),
        app_id=str(_require_value(section, "app_id")),
    )
