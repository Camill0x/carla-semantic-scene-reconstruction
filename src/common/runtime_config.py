import json
from pathlib import Path
from typing import Any, Dict, Mapping

from src.common.config import (
    CarlaConnectionConfig,
    CameraConfig,
    CollectorConfig,
    DatasetViewerConfig,
    GtAnnotationsConfig,
    LaneAnnotationsConfig,
    LidarConfig,
    LiveLaneDetInferenceConfig,
    LiveOpenPCDetInferenceConfig,
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
    )


def load_front_camera_config() -> CameraConfig:
    data = _read_runtime_config()
    section = _get_section(data, "camera_front")
    return CameraConfig(
        width=int(_require_value(section, "width")),
        height=int(_require_value(section, "height")),
        fov=float(_require_value(section, "fov")),
        x=float(_require_value(section, "x")),
        y=float(_require_value(section, "y")),
        z=float(_require_value(section, "z")),
        pitch=float(_require_value(section, "pitch")),
        yaw=float(_require_value(section, "yaw")),
        roll=float(_require_value(section, "roll")),
    )


def load_lane_annotations_config() -> LaneAnnotationsConfig:
    data = _read_runtime_config()
    section = _get_section(data, "lane_annotations")
    return LaneAnnotationsConfig(
        distance_m=float(_require_value(section, "distance_m")),
        step_m=float(_require_value(section, "step_m")),
        max_side_lanes=int(_require_value(section, "max_side_lanes")),
        projection_margin_px=float(_require_value(section, "projection_margin_px")),
        dedupe_distance_px=float(_require_value(section, "dedupe_distance_px")),
    )


def load_gt_annotations_config() -> GtAnnotationsConfig:
    data = _read_runtime_config()
    section = _get_section(data, "gt_annotations")
    return GtAnnotationsConfig(
        min_lidar_points_in_box=int(_require_value(section, "min_lidar_points_in_box")),
    )


def load_dataset_root_dir() -> Path:
    data = _read_runtime_config()
    return Path(str(_require_value(data, "dataset_root_dir")))


def build_collector_config(*, num_frames: int, every_nth: int) -> CollectorConfig:
    return CollectorConfig(
        carla=load_carla_connection_config(),
        lidar=load_lidar_config(),
        camera_front=load_front_camera_config(),
        lane_annotations=load_lane_annotations_config(),
        gt_annotations=load_gt_annotations_config(),
        dataset_root_dir=load_dataset_root_dir(),
        num_frames=num_frames,
        every_nth=every_nth,
    )


def build_live_producer_config(*, every_nth: int) -> LiveProducerConfig:
    data = _read_runtime_config()
    section = _get_section(data, "streaming", "producer")
    return LiveProducerConfig(
        carla=load_carla_connection_config(),
        lidar=load_lidar_config(),
        camera_front=load_front_camera_config(),
        lidar_bind=str(_require_value(section, "lidar_bind")),
        camera_front_bind=str(_require_value(section, "camera_front_bind")),
        state_bind=str(_require_value(section, "state_bind")),
        every_nth=every_nth,
    )


def build_live_openpcdet_inference_config(
    *,
    cfg_file: Path,
    ckpt: Path,
    score_thresh: float,
    point_stride: int,
) -> LiveOpenPCDetInferenceConfig:
    data = _read_runtime_config()
    try:
        section = _get_section(data, "streaming", "openpcdet_inference")
    except KeyError:
        section = _get_section(data, "streaming", "inference")
    return LiveOpenPCDetInferenceConfig(
        cfg_file=cfg_file,
        ckpt=ckpt,
        lidar_in=str(_require_value(section, "lidar_in")),
        zmq_out=str(_require_value(section, "zmq_out")),
        score_thresh=score_thresh,
        point_stride=point_stride,
    )


def build_live_lanedet_inference_config(
    *,
    cfg_file: Path,
    ckpt: Path,
    score_thresh: float,
) -> LiveLaneDetInferenceConfig:
    data = _read_runtime_config()
    section = _get_section(data, "streaming", "lanedet_inference")
    return LiveLaneDetInferenceConfig(
        cfg_file=cfg_file,
        ckpt=ckpt,
        camera_front_in=str(_require_value(section, "camera_front_in")),
        state_in=str(_require_value(section, "state_in")),
        zmq_out=str(_require_value(section, "zmq_out")),
        score_thresh=score_thresh,
    )


def build_live_visualizer_config(
    *,
    show_grid: bool,
) -> LiveVisualizerConfig:
    data = _read_runtime_config()
    section = _get_section(data, "streaming", "visualizer")
    objects_3d_connect = section.get("objects_3d_connect", section.get("zmq_connect"))
    return LiveVisualizerConfig(
        objects_3d_connect=str(objects_3d_connect),
        lanes_3d_connect=str(_require_value(section, "lanes_3d_connect")),
        state_connect=str(_require_value(section, "state_connect")),
        show_grid=show_grid,
        pred_line_radius=float(_require_value(section, "pred_line_radius")),
        ego_line_radius=float(_require_value(section, "ego_line_radius")),
    )


def build_dataset_viewer_config(*, show_grid: bool) -> DatasetViewerConfig:
    data = _read_runtime_config()
    section = _get_section(data, "dataset_viewer")
    return DatasetViewerConfig(
        show_grid=show_grid,
        point_radius=float(_require_value(section, "point_radius")),
        gt_line_radius=float(_require_value(section, "gt_line_radius")),
        lane_line_thickness=float(_require_value(section, "lane_line_thickness")),
    )
