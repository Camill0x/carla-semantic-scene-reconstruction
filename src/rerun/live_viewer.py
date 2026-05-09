from typing import Dict, Optional

import numpy as np

import rerun as rr
from src.common.config import LiveVisualizerConfig
from src.lanedet.prediction import Lanes3DPrediction
from src.openpcdet.prediction import Objects3DPrediction
from src.rerun.blueprints import make_live_blueprint
from src.rerun.lanes import log_prediction_lanes_3d
from src.rerun.scene3d import (
    log_ego_box,
    log_gt_boxes,
    log_points,
    log_prediction_objects_3d,
)
from src.rerun.text import log_live_status


def initialize_live_viewer(application_id: str, *, show_grid: bool) -> None:
    rr.init(application_id, spawn=True)
    rr.send_blueprint(make_live_blueprint(show_grid=show_grid))
    rr.log("world", rr.ViewCoordinates.FLU, static=True)


class LiveRenderState:
    def __init__(self, *, frame_cache_limit: int = 120) -> None:
        self.frame_cache_limit = frame_cache_limit
        self.objects_frames: Dict[int, dict] = {}
        self.lanes_frames: Dict[int, dict] = {}
        self.lidar_frames: Dict[int, dict] = {}
        self.state_frames: Dict[int, dict] = {}
        self.gt_frames: Dict[int, dict] = {}
        self.last_render_key: Optional[tuple] = None
        self.last_render_frame: Optional[int] = None

    def update_objects(self, message: dict) -> None:
        self._remember_frame(self.objects_frames, message)

    def update_lanes(self, message: dict) -> None:
        self._remember_frame(self.lanes_frames, message)

    def update_lidar(self, message: dict) -> None:
        self._remember_frame(self.lidar_frames, message)

    def update_state(self, message: dict) -> None:
        self._remember_frame(self.state_frames, message)

    def update_gt(self, message: dict) -> None:
        self._remember_frame(self.gt_frames, message)

    def _remember_frame(self, cache: Dict[int, dict], message: dict) -> None:
        cache[int(message["frame"])] = message
        while len(cache) > self.frame_cache_limit:
            del cache[min(cache)]

    def _latest_state_frame(self) -> Optional[int]:
        return max(self.state_frames) if self.state_frames else None

    def _frame_at_or_before(self, cache: Dict[int, dict], frame: int) -> Optional[dict]:
        candidates = [cached_frame for cached_frame in cache if cached_frame <= frame]
        if not candidates:
            return None
        return cache[max(candidates)]

    def render_next(self, config: LiveVisualizerConfig) -> bool:
        frame = self._latest_state_frame()
        if frame is None:
            return False
        if frame == self.last_render_frame:
            return False

        objects_for_frame = self._frame_at_or_before(self.objects_frames, frame)
        lanes_for_frame = self._frame_at_or_before(self.lanes_frames, frame)
        lidar_for_frame = self._frame_at_or_before(self.lidar_frames, frame)
        state_for_frame = self._frame_at_or_before(self.state_frames, frame)
        gt_for_frame = self._frame_at_or_before(self.gt_frames, frame)

        lanes_frame = int(lanes_for_frame["frame"]) if lanes_for_frame is not None else None
        objects_frame = int(objects_for_frame["frame"]) if objects_for_frame is not None else None
        frame_skew = abs(objects_frame - lanes_frame) if objects_frame is not None and lanes_frame is not None else None
        lidar_frame = int(lidar_for_frame["frame"]) if lidar_for_frame is not None else None
        state_frame = int(state_for_frame["frame"]) if state_for_frame is not None else None
        gt_frame = int(gt_for_frame["frame"]) if gt_for_frame is not None else None
        render_key = (frame, objects_frame, lanes_frame, lidar_frame, state_frame, gt_frame)
        if render_key == self.last_render_key:
            return False

        points = lidar_for_frame["points"] if lidar_for_frame is not None else np.zeros((0, 4), dtype=np.float32)
        ego_payload = state_for_frame.get("ego", {}) if state_for_frame is not None else {}
        ego_box = np.asarray(ego_payload.get("box", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        gt_boxes = gt_for_frame["gt_boxes"] if gt_for_frame is not None else np.zeros((0, 7), dtype=np.float32)
        gt_names = gt_for_frame["gt_names"] if gt_for_frame is not None else []
        objects_3d = objects_for_frame["objects_3d"] if objects_for_frame is not None else Objects3DPrediction.empty()
        lanes_3d = lanes_for_frame["lanes_3d"] if lanes_for_frame is not None else Lanes3DPrediction.empty()
        num_lanes = len(lanes_3d)

        rr.set_time("frame", sequence=frame)

        log_points(points, point_radius=config.point_radius, visible=points.shape[0] > 0)
        log_gt_boxes(
            gt_boxes,
            gt_names,
            line_radius=config.gt_line_radius,
            visible=gt_boxes.shape[0] > 0,
        )
        log_ego_box(ego_box, line_radius=config.gt_line_radius)
        log_prediction_objects_3d(objects_3d, line_radius=config.pred_line_radius)
        log_prediction_lanes_3d(
            lanes_3d,
            line_radius=config.pred_line_radius,
        )
        log_live_status(
            frame=frame,
            num_points=int(points.shape[0]),
            num_gt=int(gt_boxes.shape[0]),
            num_pred=len(objects_3d),
            num_lanes=num_lanes,
            objects_frame=objects_frame,
            lanes_frame=lanes_frame,
            frame_skew=frame_skew,
        )
        self.last_render_key = render_key
        self.last_render_frame = frame
        return True
