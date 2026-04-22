from dataclasses import dataclass


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
    ego_bbox_padding: float

    def __post_init__(self) -> None:
        if self.max_range <= 0.0:
            raise ValueError("LiDAR range must be > 0")
        if self.channels <= 0:
            raise ValueError("LiDAR channels must be > 0")
        if self.points_per_second <= 0:
            raise ValueError("LiDAR points_per_second must be > 0")
        if self.ego_bbox_padding < 0.0:
            raise ValueError("LiDAR ego_bbox_padding must be >= 0")


@dataclass(frozen=True)
class CollectorConfig:
    carla: CarlaConnectionConfig
    lidar: LidarConfig
    output_dir: str
    num_frames: int
    every_nth: int

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("Collector output_dir must not be empty")
        if self.num_frames <= 0:
            raise ValueError("Collector num_frames must be > 0")
        if self.every_nth <= 0:
            raise ValueError("Collector every_nth must be >= 1")

    @property
    def max_range(self) -> float:
        return self.lidar.max_range

    @property
    def ego_bbox_padding(self) -> float:
        return self.lidar.ego_bbox_padding

@dataclass(frozen=True)
class LiveProducerConfig:
    carla: CarlaConnectionConfig
    lidar: LidarConfig
    zmq_bind: str
    every_nth: int
    with_gt: bool

    def __post_init__(self) -> None:
        if not self.zmq_bind:
            raise ValueError("Producer zmq_bind must not be empty")
        if self.every_nth <= 0:
            raise ValueError("Producer every_nth must be >= 1")

    @property
    def max_range(self) -> float:
        return self.lidar.max_range

    @property
    def ego_bbox_padding(self) -> float:
        return self.lidar.ego_bbox_padding

@dataclass(frozen=True)
class LiveInferenceConfig:
    cfg_file: str
    ckpt: str
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
    app_id: str

    def __post_init__(self) -> None:
        if not self.zmq_connect:
            raise ValueError("Visualizer zmq_connect must not be empty")
        if not self.app_id:
            raise ValueError("Visualizer app_id must not be empty")
        if self.point_radius <= 0.0:
            raise ValueError("Visualizer point_radius must be > 0")
        if self.pred_line_radius <= 0.0:
            raise ValueError("Visualizer pred_line_radius must be > 0")
        if self.gt_line_radius <= 0.0:
            raise ValueError("Visualizer gt_line_radius must be > 0")
