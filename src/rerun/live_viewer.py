import rerun as rr
import rerun.blueprint as rrb


def make_live_blueprint(show_grid: bool):
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="Live Scene",
                line_grid=rrb.LineGrid3D(visible=show_grid),
                eye_controls=rrb.EyeControls3D(
                    position=(-22.0, 0.0, 10.5),
                    look_target=(8.0, 0.0, 0.0),
                    eye_up=(0.0, 0.0, 1.0),
                    speed=18.0,
                ),
            ),
            rrb.Vertical(
                rrb.TextDocumentView(origin="/status", name="Status"),
                rrb.TextDocumentView(origin="/legend", name="Legend"),
                row_shares=[0.65, 0.35],
            ),
            column_shares=[0.82, 0.18],
        ),
        collapse_panels=False,
    )


def initialize_live_viewer(application_id: str, *, show_grid: bool) -> None:
    rr.init(application_id, spawn=True)
    rr.send_blueprint(make_live_blueprint(show_grid=show_grid))
    rr.log("world", rr.ViewCoordinates.FLU, static=True)


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
