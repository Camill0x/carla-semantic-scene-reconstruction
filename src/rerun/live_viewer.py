import numpy as np

import rerun as rr
from src.common.typing_aliases import JsonDict
from src.rerun.blueprints import make_live_blueprint
from src.rerun.lanes import log_prediction_lanes_3d
from src.rerun.scene3d import log_ego_box, log_prediction_objects_3d
from src.rerun.text import log_live_status


def initialize_live_viewer(application_id: str, *, show_grid: bool) -> None:
    rr.init(application_id, spawn=True)
    rr.send_blueprint(make_live_blueprint(show_grid=show_grid))
    rr.log("world", rr.ViewCoordinates.FLU, static=True)


def render_live_scene(
    scene: JsonDict,
    *,
    ego_line_radius: float,
    pred_line_radius: float,
) -> None:
    source_frames = scene.get("source_frames", {})
    frame = int(scene["frame"])
    objects_frame = source_frames.get("objects_3d")
    lanes_frame = source_frames.get("lanes_3d")
    ego_payload = scene.get("ego", {})
    ego_box = np.asarray(ego_payload.get("box", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    objects_3d = scene["objects_3d"]
    lanes_3d = scene["lanes_3d"]

    rr.set_time("frame", sequence=frame)
    log_ego_box(ego_box, line_radius=ego_line_radius)
    log_prediction_objects_3d(objects_3d, line_radius=pred_line_radius)
    log_prediction_lanes_3d(lanes_3d, line_radius=pred_line_radius)
    log_live_status(
        frame=frame,
        latency_ms=float(scene["latency_ms"]),
        transfer_bytes=int(scene["transfer_bytes"]),
        num_obj=len(objects_3d) if objects_frame is not None else None,
        num_lanes=len(lanes_3d) if lanes_frame is not None else None,
    )
