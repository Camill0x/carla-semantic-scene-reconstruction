from typing import Dict, List

CLASS_COLORS = {
    "car": [0.95, 0.20, 0.20],
    "truck": [0.75, 0.10, 0.10],
    "bus": [1.00, 0.45, 0.25],
    "motorcycle": [1.00, 0.70, 0.10],
    "bicycle": [0.95, 0.90, 0.20],
    "pedestrian": [0.15, 0.85, 0.25],
}

STATIC_COLOR_SCALE = 0.75
EGO_COLOR = [0.25, 0.85, 1.00]


def get_object_color(obj: Dict) -> List[float]:
    base = CLASS_COLORS.get(obj["class_name"], [1.0, 0.0, 1.0])
    if obj.get("source") == "static_level_bbox":
        return [max(0.0, min(1.0, c * STATIC_COLOR_SCALE)) for c in base]
    return base
