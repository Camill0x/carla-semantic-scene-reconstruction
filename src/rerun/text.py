from typing import Optional

import rerun as rr
from src.carla.dataset.reader import DatasetFrame


def log_dataset_status(
    frame: DatasetFrame,
    *,
    num_predicted_objects: Optional[int] = None,
    num_predicted_lanes: Optional[int] = None,
) -> None:
    """Log the offline dataset status panel to Rerun."""
    lines = [
        "## Status",
        "",
        f"- Run frame: {frame.meta.get('frame_index', -1)}",
        f"- Sim frame: {frame.meta.get('sim_frame', -1)}",
        f"- Timestamp: {frame.meta.get('timestamp', -1.0):.3f}",
        f"- Points: {int(frame.points.shape[0])}",
        f"- Objects GT: {len(frame.objects)}",
        f"- Lanes GT: {len(frame.lanes)}",
    ]
    if num_predicted_objects is not None:
        lines.append(f"- Objects predicted: {num_predicted_objects}")
    if num_predicted_lanes is not None:
        lines.append(f"- Lanes predicted: {num_predicted_lanes}")
    rr.log("status", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN))


def log_live_status(
    *,
    frame: int,
    latency_ms: float,
    transfer_bytes: int,
    num_obj: Optional[int] = None,
    num_lanes: Optional[int] = None,
) -> None:
    """Log the live streaming status panel to Rerun."""
    lines = [
        "## Status",
        "",
        f"- Frame id: {frame}",
        f"- Latency: {latency_ms:.2f} ms",
        f"- Scene transfer size: {transfer_bytes / 1024.0:.2f} KiB",
    ]
    if num_obj is not None:
        lines.append(f"- Objects predicted: {num_obj}")
    if num_lanes is not None:
        lines.append(f"- Lanes predicted: {num_lanes}")
    rr.log("status", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN))


def log_legend() -> None:
    """Log the viewer legend panel to Rerun."""
    lines = [
        "## Colors",
        "",
        "- Ego box: cyan",
        "- Car: red",
        "- Truck: dark red",
        "- Bus: orange",
        "- Motorcycle: amber",
        "- Bicycle: yellow",
        "- Pedestrian: green",
    ]
    rr.log("legend", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)
