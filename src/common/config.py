from dataclasses import dataclass


@dataclass(frozen=True)
class CollectorConfig:
    max_range: float
    ego_bbox_padding: float
    output_dir: str
    num_frames: int
    every_nth: int
