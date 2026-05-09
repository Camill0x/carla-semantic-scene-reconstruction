from typing import Any, Dict, Mapping

import numpy as np


def transform_from_meta(sensor_payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "location": dict(sensor_payload["location"]),
        "rotation": dict(sensor_payload["rotation"]),
    }


def state_frame_from_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    lidar_meta = meta["lidar"]
    camera_meta = meta["front_camera"]
    hero_meta = meta.get("hero", {})
    hero_transform = hero_meta.get("transform", {}) if isinstance(hero_meta, Mapping) else {}
    hero_location = hero_transform.get("location", {}) if isinstance(hero_transform, Mapping) else {}

    lidar_location = lidar_meta["location"]
    ground_z = float(hero_location.get("z", 0.0)) - float(lidar_location["z"])

    return {
        "frame": int(meta.get("frame_index", -1)),
        "timestamp": float(meta.get("timestamp", -1.0)),
        "ego": {},
        "lidar": {
            "transform": transform_from_meta(lidar_meta),
            "ground_z": ground_z,
        },
        "camera_front": {
            "transform": transform_from_meta(camera_meta),
            "fov": float(camera_meta["fov"]),
        },
    }


def camera_frame_shape(image_rgb: np.ndarray) -> Dict[str, Any]:
    return {
        "camera_front": {
            "height": int(image_rgb.shape[0]),
            "width": int(image_rgb.shape[1]),
        },
    }
