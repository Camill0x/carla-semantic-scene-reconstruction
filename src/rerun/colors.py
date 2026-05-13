from typing import Iterable, Tuple

CLASS_COLORS = {
    "car": (242, 51, 51, 235),
    "truck": (191, 26, 26, 235),
    "bus": (255, 115, 64, 235),
    "motorcycle": (255, 179, 26, 235),
    "bicycle": (242, 230, 51, 235),
    "pedestrian": (38, 217, 64, 235),
}

GT_COLOR = (77, 163, 255, 210)
EGO_COLOR = (64, 255, 255, 235)
LANE_LEFT_COLOR = (0, 255, 0, 255)
LANE_RIGHT_COLOR = (0, 200, 255, 255)


def prediction_colors(pred_names: Iterable[str]) -> list:
    return [CLASS_COLORS.get(str(name), (255, 255, 255, 235)) for name in pred_names]


def lane_color(lane: dict) -> Tuple[int, int, int, int]:
    if str(lane.get("side", "")) == "left":
        return LANE_LEFT_COLOR
    if str(lane.get("side", "")) == "right":
        return LANE_RIGHT_COLOR
    return (255, 255, 255, 255)
