import rerun as rr
from src.carla.dataset.reader import DatasetFrame


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


def log_live_status(
    *,
    frame: int,
    num_points: int,
    num_gt: int,
    num_pred: int,
    score_thresh: float | None = None,
) -> None:
    lines = [
        "## Status",
        "",
        f"- Frame: {frame}",
        f"- Points: {num_points}",
        f"- GT boxes: {num_gt}",
        f"- Pred boxes: {num_pred}",
    ]
    if score_thresh is not None:
        lines.append(f"- Score threshold: {score_thresh:.2f}")
    rr.log("status", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN))


def log_legend() -> None:
    lines = [
        "## Colors",
        "",
        "- GT boxes: blue",
        "- Ego box: cyan",
        "- Car: red",
        "- Truck: dark red",
        "- Bus: orange",
        "- Motorcycle: amber",
        "- Bicycle: yellow",
        "- Pedestrian: green",
        "- Points: intensity shaded",
    ]
    rr.log("legend", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)
