from src.common.constants import NUSCENES_LIKE_CLASSES

OPENPCDET_PRESETS = [
    "transfusion-ft",
    "transfusion-zeroshot",
    "centerpoint-pp-ft",
    "centerpoint-pp-zeroshot",
]

CLASS_FILTERS = {
    "carla_nuscenes6": tuple(NUSCENES_LIKE_CLASSES),
}
