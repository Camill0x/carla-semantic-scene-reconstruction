from typing import List

import numpy as np

import rerun as rr
import rerun.blueprint as rrb
from src.carla.dataset.reader import DatasetFrame
from src.common.config import DatasetViewerConfig
from src.rerun.scene3d import (
    log_ego_box,
    log_gt_boxes,
    log_lane_annotations_3d,
    log_points,
)

LANE_LEFT_COLOR = (0, 255, 0, 255)
LANE_RIGHT_COLOR = (0, 200, 255, 255)


def make_dataset_blueprint(show_grid: bool):
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/camera/front", name="Front Camera"),
                rrb.Spatial3DView(
                    origin="/world",
                    name="LiDAR",
                    line_grid=rrb.LineGrid3D(visible=show_grid),
                    eye_controls=rrb.EyeControls3D(
                        position=(-22.0, 0.0, 10.5),
                        look_target=(8.0, 0.0, 0.0),
                        eye_up=(0.0, 0.0, 1.0),
                        speed=18.0,
                    ),
                ),
                row_shares=[0.58, 0.42],
            ),
            rrb.TextDocumentView(origin="/status", name="Status"),
            column_shares=[0.88, 0.12],
        ),
        collapse_panels=False,
    )


def initialize_dataset_viewer(config: DatasetViewerConfig) -> None:
    rr.init("carla_dataset_viewer", spawn=True)
    rr.send_blueprint(make_dataset_blueprint(show_grid=config.show_grid))
    rr.log("world", rr.ViewCoordinates.FLU, static=True)


def _lane_color(lane: dict) -> tuple[int, int, int, int]:
    if str(lane.get("side", "")) == "left":
        return LANE_LEFT_COLOR
    if str(lane.get("side", "")) == "right":
        return LANE_RIGHT_COLOR
    return (255, 255, 255, 255)


def log_lane_annotations_2d(lanes: List[dict], *, line_thickness: float) -> None:
    strips = []
    colors = []
    labels = []

    for lane in lanes:
        points = lane.get("points", [])
        if len(points) < 2:
            continue
        strips.append(np.asarray(points, dtype=np.float32))
        colors.append(_lane_color(lane))
        labels.append(f"lane {lane.get('lane_id', '?')} {lane.get('side', '?')} {lane.get('marking_type', '?')}")

    if not strips:
        rr.log("camera/front/lanes", rr.LineStrips2D(strips=[]))
        return

    rr.log(
        "camera/front/lanes",
        rr.LineStrips2D(
            strips=strips,
            colors=colors,
            radii=line_thickness,
            labels=labels,
            show_labels=False,
        ),
    )


def clamp_lane_annotations_to_image(lanes: List[dict], image_width: int, image_height: int) -> List[dict]:
    clamped_lanes: List[dict] = []
    max_x = float(max(image_width - 1, 0))
    max_y = float(max(image_height - 1, 0))

    for lane in lanes:
        points = lane.get("points", [])
        if len(points) < 2:
            continue

        clamped_points = []
        for point in points:
            if len(point) != 2:
                continue
            clamped_points.append(
                [
                    float(np.clip(float(point[0]), 0.0, max_x)),
                    float(np.clip(float(point[1]), 0.0, max_y)),
                ]
            )

        if len(clamped_points) < 2:
            continue

        clamped_lanes.append({**lane, "points": clamped_points})

    return clamped_lanes


def log_dataset_status(frame: DatasetFrame) -> None:
    lines = [
        "## Status",
        "",
        f"- Run frame: {frame.meta.get('frame_index', -1)}",
        f"- Sim frame: {frame.meta.get('sim_frame', -1)}",
        f"- Timestamp: {frame.meta.get('timestamp', -1.0):.3f}",
        f"- Points: {int(frame.points.shape[0])}",
        f"- GT Objects: {len(frame.objects)}",
        f"- Lanes: {len(frame.lanes)}",
    ]
    rr.log("status", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN))


def log_dataset_frame(frame: DatasetFrame, config: DatasetViewerConfig) -> None:
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
