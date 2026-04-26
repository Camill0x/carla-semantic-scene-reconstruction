from src.common.constants import NUSCENES_LIKE_CLASSES

DEFAULT_DATASET_NAME = "default"
DATASET_FAMILY = "carla_nuscenes6"
OPENPCDET_RESULTS_DIR = "openpcdet"
OPENPCDET_THIRD_PARTY_DIR = ("third_party", "OpenPCDet")

OPENPCDET_PRESETS = {
    "transfusion-ft": "carla_nuscenes6_transfusion_ft.yaml",
    "transfusion-zeroshot": "carla_nuscenes6_transfusion_zeroshot.yaml",
    "centerpoint-ft": "carla_nuscenes6_centerpoint_pp_ft.yaml",
    "centerpoint-zeroshot": "carla_nuscenes6_centerpoint_pp_zeroshot.yaml",
}

CLASS_FILTERS = {
    "nuscenes6": tuple(NUSCENES_LIKE_CLASSES),
}
