from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CarlaConnectionConfig:
    host: str
    port: int

    def __post_init__(self) -> None:
        """Validate the CarlaConnectionConfig configuration after initialization."""
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
        """Validate the LidarConfig configuration after initialization."""
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
        """Validate the CameraConfig configuration after initialization."""
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
        """Validate the LaneAnnotationsConfig configuration after initialization."""
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
        """Validate the GtAnnotationsConfig configuration after initialization."""
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
        """Validate the CollectorConfig configuration after initialization."""
        if not self.dataset_root_dir:
            raise ValueError("Collector dataset_root_dir must not be empty")
        if self.num_frames <= 0:
            raise ValueError("Collector num_frames must be > 0")
        if self.every_nth <= 0:
            raise ValueError("Collector every_nth must be >= 1")


@dataclass(frozen=True)
class DatasetViewerConfig:
    show_grid: bool
    point_radius: float
    gt_line_radius: float
    lane_line_thickness: float

    def __post_init__(self) -> None:
        """Validate the DatasetViewerConfig configuration after initialization."""
        if self.point_radius <= 0.0:
            raise ValueError("Dataset viewer point_radius must be > 0")
        if self.gt_line_radius <= 0.0:
            raise ValueError("Dataset viewer gt_line_radius must be > 0")
        if self.lane_line_thickness <= 0.0:
            raise ValueError("Dataset viewer lane_line_thickness must be > 0")


@dataclass(frozen=True)
class StreamingCommonConfig:
    prefix: str
    poll_interval_ms: int
    frame_buffer_size_bytes: int
    objects_buffer_size_bytes: int
    lanes_buffer_size_bytes: int

    def __post_init__(self) -> None:
        """Validate the StreamingCommonConfig configuration after initialization."""
        if not self.prefix:
            raise ValueError("Streaming prefix must not be empty")
        if self.poll_interval_ms < 0:
            raise ValueError("Streaming poll_interval_ms must be >= 0")
        if self.frame_buffer_size_bytes <= 0:
            raise ValueError("Streaming frame_buffer_size_bytes must be > 0")
        if self.objects_buffer_size_bytes <= 0:
            raise ValueError("Streaming objects_buffer_size_bytes must be > 0")
        if self.lanes_buffer_size_bytes <= 0:
            raise ValueError("Streaming lanes_buffer_size_bytes must be > 0")


@dataclass(frozen=True)
class StreamingProducerConfig:
    common: StreamingCommonConfig
    carla: CarlaConnectionConfig
    lidar: LidarConfig
    camera_front: CameraConfig
    every_nth: int
    sensor_slots: int
    lidar_slot_capacity_bytes: int

    def __post_init__(self) -> None:
        """Validate the StreamingProducerConfig configuration after initialization."""
        if self.every_nth <= 0:
            raise ValueError("Streaming producer every_nth must be >= 1")
        if self.sensor_slots <= 0:
            raise ValueError("Streaming producer sensor_slots must be > 0")
        if self.lidar_slot_capacity_bytes <= 0:
            raise ValueError("Streaming producer lidar_slot_capacity_bytes must be > 0")


@dataclass(frozen=True)
class StreamingOpenPCDetInferenceConfig:
    common: StreamingCommonConfig
    cfg_file: Path
    ckpt: Path
    score_thresh: float
    point_stride: int

    def __post_init__(self) -> None:
        """Validate the StreamingOpenPCDetInferenceConfig configuration after initialization."""
        if not self.cfg_file:
            raise ValueError("Streaming OpenPCDet cfg_file must not be empty")
        if not self.ckpt:
            raise ValueError("Streaming OpenPCDet ckpt must not be empty")
        if self.score_thresh < 0:
            raise ValueError("Streaming OpenPCDet score_thresh must be >= 0")
        if self.point_stride < 1:
            raise ValueError("Streaming OpenPCDet point_stride must be > 1")


@dataclass(frozen=True)
class StreamingLaneDetInferenceConfig:
    common: StreamingCommonConfig
    cfg_file: Path
    ckpt: Path
    score_thresh: float

    def __post_init__(self) -> None:
        """Validate the StreamingLaneDetInferenceConfig configuration after initialization."""
        if not self.cfg_file:
            raise ValueError("Streaming LaneDet cfg_file must not be empty")
        if not self.ckpt:
            raise ValueError("Streaming LaneDet ckpt must not be empty")
        if self.score_thresh < 0:
            raise ValueError("Streaming LaneDet score_thresh must be >= 0")


@dataclass(frozen=True)
class StreamingAggregatorConfig:
    common: StreamingCommonConfig
    scene_bind: str

    def __post_init__(self) -> None:
        """Validate the StreamingAggregatorConfig configuration after initialization."""
        if not self.scene_bind:
            raise ValueError("Streaming aggregator scene_bind must not be empty")


@dataclass(frozen=True)
class StreamingVisualizerConfig:
    scene_connect: str
    show_grid: bool
    pred_line_radius: float
    ego_line_radius: float

    def __post_init__(self) -> None:
        """Validate the StreamingVisualizerConfig configuration after initialization."""
        if not self.scene_connect:
            raise ValueError("Streaming visualizer scene_connect must not be empty")
        if self.pred_line_radius <= 0.0:
            raise ValueError("Streaming visualizer pred_line_radius must be > 0")
        if self.ego_line_radius <= 0.0:
            raise ValueError("Streaming visualizer ego_line_radius must be > 0")
