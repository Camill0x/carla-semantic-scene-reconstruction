import rerun as rr
from src.carla.dataset.reader import DatasetFrame
from src.common.config import DatasetViewerConfig
from src.rerun.blueprints import make_dataset_blueprint
from src.rerun.lanes import (
    clamp_lane_annotations_to_image,
    log_lane_annotations_2d,
    log_lane_annotations_3d,
)
from src.rerun.scene3d import log_ego_box, log_gt_boxes, log_points
from src.rerun.text import log_dataset_status


def initialize_dataset_viewer(config: DatasetViewerConfig) -> None:
    """Initialize the Rerun dataset viewer and its default blueprint."""
    rr.init("carla_dataset_viewer", spawn=True)
    rr.send_blueprint(make_dataset_blueprint(show_grid=config.show_grid))
    rr.log("world", rr.ViewCoordinates.FLU, static=True)


def log_dataset_frame(frame: DatasetFrame, config: DatasetViewerConfig) -> None:
    """Log one recorded dataset frame into the Rerun dataset viewer."""
    frame_index = int(frame.meta.get("frame_index", 0))
    rr.set_time("frame_idx", sequence=frame_index)

    log_points(frame.points, point_radius=config.point_radius, visible=True)
    log_gt_boxes(
        frame.gt_boxes,
        frame.gt_names,
        line_radius=config.gt_line_radius,
        visible=True,
    )
    log_ego_box(
        frame.ego_box,
        line_radius=config.gt_line_radius,
    )
    log_lane_annotations_3d(
        frame.lanes,
        line_radius=config.gt_line_radius,
    )
    rr.log("camera/front/image", rr.Image(frame.image_rgb))
    clamped_lanes = clamp_lane_annotations_to_image(
        frame.lanes,
        image_width=int(frame.image_rgb.shape[1]),
        image_height=int(frame.image_rgb.shape[0]),
    )
    log_lane_annotations_2d(clamped_lanes, line_thickness=config.lane_line_thickness)
    log_dataset_status(frame)
