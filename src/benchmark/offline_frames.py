from typing import Any, Dict, Mapping

import numpy as np

from src.streaming.messages import build_camera_frame_message


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
        "schema": "state_frame",
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


def camera_frame_from_image_rgb_and_meta(image_rgb: np.ndarray, meta: Mapping[str, Any]) -> Dict[str, Any]:
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    return build_camera_frame_message(
        frame=int(meta.get("frame_index", -1)),
        timestamp=float(meta.get("timestamp", -1.0)),
        camera_front_image=image_bgr,
    )
