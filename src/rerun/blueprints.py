from typing import Any

import rerun.blueprint as rrb


def make_dataset_blueprint(show_grid: bool) -> Any:
    """Build the default Rerun blueprint for offline dataset playback."""
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


def make_live_blueprint(show_grid: bool) -> Any:
    """Build the default Rerun blueprint for live scene playback."""
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
