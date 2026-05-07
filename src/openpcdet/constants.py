from src.common.constants import NUSCENES_LIKE_CLASSES

OPENPCDET_PRESETS = {
    "cn6-transfusion-ft": ("carla_nuscenes6", "carla_nuscenes6_transfusion_ft.yaml"),
    "cn6-transfusion-zeroshot": ("carla_nuscenes6", "carla_nuscenes6_transfusion_zeroshot.yaml"),
    "cn6-centerpoint-pp-ft": ("carla_nuscenes6", "carla_nuscenes6_centerpoint_pp_ft.yaml"),
    "cn6-centerpoint-pp-zeroshot": ("carla_nuscenes6", "carla_nuscenes6_centerpoint_pp_zeroshot.yaml"),
}

CLASS_FILTERS = {
    "carla_nuscenes6": tuple(NUSCENES_LIKE_CLASSES),
}
