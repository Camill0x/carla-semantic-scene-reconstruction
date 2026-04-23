import rerun as rr
from src.rerun.blueprints import make_live_blueprint
from src.rerun.text import log_legend, log_live_status


def initialize_live_viewer(application_id: str, *, show_grid: bool) -> None:
    rr.init(application_id, spawn=True)
    rr.send_blueprint(make_live_blueprint(show_grid=show_grid))
    rr.log("world", rr.ViewCoordinates.FLU, static=True)
