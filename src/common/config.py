from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CarlaConnectionConfig:
    host: str
    port: int

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("CARLA host must not be empty")
        if self.port <= 0:
            raise ValueError("CARLA port must be > 0")


@dataclass(frozen=True)
class LidarConfig:
    max_range: float
    channels: int
    points_per_second: int
    upper_fov: float
    lower_fov: float

    def __post_init__(self) -> None:
        if self.max_range <= 0.0:
            raise ValueError("LiDAR range must be > 0")
        if self.channels <= 0:
            raise ValueError("LiDAR channels must be > 0")
        if self.points_per_second <= 0:
            raise ValueError("LiDAR points_per_second must be > 0")


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fov: float
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("Camera width must be > 0")
        if self.height <= 0:
            raise ValueError("Camera height must be > 0")
        if self.fov <= 0.0:
            raise ValueError("Camera fov must be > 0")


@dataclass(frozen=True)
class LaneAnnotationsConfig:
    distance_m: float
    step_m: float
    max_side_lanes: int
    projection_margin_px: float
    dedupe_distance_px: float

    def __post_init__(self) -> None:
        if self.distance_m <= 0.0:
            raise ValueError("Lane distance_m must be > 0")
        if self.step_m <= 0.0:
            raise ValueError("Lane step_m must be > 0")
        if self.max_side_lanes < 0:
            raise ValueError("Lane max_side_lanes must be >= 0")
        if self.projection_margin_px < 0.0:
            raise ValueError("Lane projection_margin_px must be >= 0")
        if self.dedupe_distance_px < 0.0:
            raise ValueError("Lane dedupe_distance_px must be >= 0")


@dataclass(frozen=True)
class GtAnnotationsConfig:
    min_lidar_points_in_box: int

    def __post_init__(self) -> None:
        if self.min_lidar_points_in_box < 0:
            raise ValueError("GT annotations min_lidar_points_in_box must be >= 0")


@dataclass(frozen=True)
class CollectorConfig:
    carla: CarlaConnectionConfig
    lidar: LidarConfig
    camera_front: CameraConfig
    lane_annotations: LaneAnnotationsConfig
    gt_annotations: GtAnnotationsConfig
    dataset_root_dir: Path
    num_frames: int
    every_nth: int

    def __post_init__(self) -> None:
        if not self.dataset_root_dir:
            raise ValueError("Collector dataset_root_dir must not be empty")
        if self.num_frames <= 0:
            raise ValueError("Collector num_frames must be > 0")
        if self.every_nth <= 0:
            raise ValueError("Collector every_nth must be >= 1")

@dataclass(frozen=True)
class LiveProducerConfig:
    carla: CarlaConnectionConfig
    lidar: LidarConfig
    gt_annotations: GtAnnotationsConfig
    zmq_bind: str
    every_nth: int
    with_gt: bool

    def __post_init__(self) -> None:
        if not self.zmq_bind:
            raise ValueError("Producer zmq_bind must not be empty")
        if self.every_nth <= 0:
            raise ValueError("Producer every_nth must be >= 1")

@dataclass(frozen=True)
class LiveInferenceConfig:
    cfg_file: Path
    ckpt: Path
    zmq_in: str
    zmq_out: str
    score_thresh: float
    point_stride: int

    def __post_init__(self) -> None:
        if not self.cfg_file:
            raise ValueError("Inference cfg_file must not be empty")
        if not self.ckpt:
            raise ValueError("Inference ckpt must not be empty")
        if not self.zmq_in:
            raise ValueError("Inference zmq_in must not be empty")
        if not self.zmq_out:
            raise ValueError("Inference zmq_out must not be empty")
        if self.point_stride < 1:
            raise ValueError("Inference point_stride must be >= 1")


@dataclass(frozen=True)
class LiveVisualizerConfig:
    zmq_connect: str
    show_grid: bool
    hide_points: bool
    hide_gt: bool
    point_radius: float
    pred_line_radius: float
    gt_line_radius: float

    def __post_init__(self) -> None:
        if not self.zmq_connect:
            raise ValueError("Visualizer zmq_connect must not be empty")
        if self.point_radius <= 0.0:
            raise ValueError("Visualizer point_radius must be > 0")
        if self.pred_line_radius <= 0.0:
            raise ValueError("Visualizer pred_line_radius must be > 0")
        if self.gt_line_radius <= 0.0:
            raise ValueError("Visualizer gt_line_radius must be > 0")


@dataclass(frozen=True)
class DatasetViewerConfig:
    show_grid: bool
    point_radius: float
    gt_line_radius: float
    lane_line_thickness: float

    def __post_init__(self) -> None:
        if self.point_radius <= 0.0:
            raise ValueError("Dataset viewer point_radius must be > 0")
        if self.gt_line_radius <= 0.0:
            raise ValueError("Dataset viewer gt_line_radius must be > 0")
        if self.lane_line_thickness <= 0.0:
            raise ValueError("Dataset viewer lane_line_thickness must be > 0")
