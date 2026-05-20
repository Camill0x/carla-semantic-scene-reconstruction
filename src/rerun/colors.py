from typing import Dict, Iterable, List, Mapping, Tuple, TypeAlias

RGBA: TypeAlias = Tuple[int, int, int, int]

CLASS_COLORS: Dict[str, RGBA] = {
    "car": (242, 51, 51, 235),
    "truck": (191, 26, 26, 235),
    "bus": (255, 115, 64, 235),
    "motorcycle": (255, 179, 26, 235),
    "bicycle": (242, 230, 51, 235),
    "pedestrian": (38, 217, 64, 235),
}

GT_COLOR: RGBA = (77, 163, 255, 210)
EGO_COLOR: RGBA = (64, 255, 255, 235)
LANE_LEFT_COLOR: RGBA = (0, 255, 0, 255)
LANE_RIGHT_COLOR: RGBA = (0, 200, 255, 255)


def prediction_colors(pred_names: Iterable[str]) -> List[RGBA]:
    """Return display colors for the provided prediction class names."""
    return [CLASS_COLORS.get(str(name), (255, 255, 255, 235)) for name in pred_names]


def lane_color(lane: Mapping[str, object]) -> RGBA:
    """Return the display color for a lane annotation or prediction."""
    if str(lane.get("side", "")) == "left":
        return LANE_LEFT_COLOR
    if str(lane.get("side", "")) == "right":
        return LANE_RIGHT_COLOR
    return (255, 255, 255, 255)
