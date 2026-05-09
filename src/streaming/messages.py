from typing import Any, Dict, Mapping

import numpy as np

from src.lanedet.prediction import Lanes3DPrediction
from src.openpcdet.prediction import Objects3DPrediction


def build_lidar_frame_message(
    *,
    frame: int,
    timestamp: float,
    points: np.ndarray,
) -> Dict[str, Any]:
    points_array = np.asarray(points, dtype=np.float32)
    return {
        "schema": "lidar_frame",
        "frame": int(frame),
        "timestamp": float(timestamp),
        "lidar": {
            "points": points_array,
        },
    }


def parse_lidar_frame_message(message: Mapping[str, Any]) -> Dict[str, Any]:
    frame_id = int(message.get("frame", -1))
    lidar_payload = message.get("lidar", {})
    if not isinstance(lidar_payload, Mapping):
        lidar_payload = {}

    points = np.asarray(lidar_payload.get("points", message.get("points")), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Invalid points shape: {points.shape}")

    lidar = {
        **{key: value for key, value in lidar_payload.items() if key != "points"},
        "points": points,
    }
    return {
        "schema": str(message.get("schema", "lidar_frame")),
        "frame": frame_id,
        "timestamp": float(message.get("timestamp", -1.0)),
        "lidar": lidar,
        "points": points,
    }


def build_camera_frame_message(
    *,
    frame: int,
    timestamp: float,
    camera_front_image: np.ndarray,
) -> Dict[str, Any]:
    image_array = np.asarray(camera_front_image, dtype=np.uint8)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Invalid camera_front image shape: {image_array.shape}")

    return {
        "schema": "camera_frame",
        "frame": int(frame),
        "timestamp": float(timestamp),
        "camera_front": {
            "image": image_array,
            "height": int(image_array.shape[0]),
            "width": int(image_array.shape[1]),
        },
    }


def parse_camera_frame_message(message: Mapping[str, Any]) -> Dict[str, Any]:
    camera_payload = message.get("camera_front")
    if not isinstance(camera_payload, Mapping):
        raise ValueError("Missing camera_front payload")
    camera_front = dict(camera_payload)
    image = np.asarray(camera_front["image"], dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid camera_front image shape: {image.shape}")
    camera_front["image"] = image
    camera_front["height"] = int(camera_front.get("height", image.shape[0]))
    camera_front["width"] = int(camera_front.get("width", image.shape[1]))
    return {
        "schema": str(message.get("schema", "camera_frame")),
        "frame": int(message.get("frame", -1)),
        "timestamp": float(message.get("timestamp", -1.0)),
        "camera_front": camera_front,
    }


def build_state_frame_message(
    *,
    frame: int,
    timestamp: float,
    ego_box: np.ndarray,
    lidar_metadata: Mapping[str, Any],
    camera_front_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "frame": int(frame),
        "timestamp": float(timestamp),
        "ego": {
            "box": np.asarray(ego_box, dtype=np.float32),
        },
        "lidar": {
            "transform": lidar_metadata["transform"],
            "ground_z": float(lidar_metadata["ground_z"]),
        },
        "camera_front": {
            "transform": camera_front_metadata["transform"],
            "fov": float(camera_front_metadata["fov"]),
        },
    }


def parse_state_frame_message(message: Mapping[str, Any]) -> Dict[str, Any]:
    ego = message.get("ego")
    lidar = message.get("lidar")
    camera_front = message.get("camera_front")
    if not isinstance(ego, Mapping):
        ego = {}
    if not isinstance(lidar, Mapping):
        raise ValueError("Missing lidar state payload")
    if not isinstance(camera_front, Mapping):
        raise ValueError("Missing camera_front state payload")
    ego_box = np.asarray(ego.get("box", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    return {
        "frame": int(message.get("frame", -1)),
        "timestamp": float(message.get("timestamp", -1.0)),
        "ego": {"box": ego_box},
        "lidar": dict(lidar),
        "camera_front": dict(camera_front),
    }
def build_objects_3d_frame_message(
    *,
    lidar_message: Mapping[str, Any],
    objects_3d: Objects3DPrediction,
) -> Dict[str, Any]:
    return {
        "schema": "objects_3d_frame",
        "frame": int(lidar_message.get("frame", -1)),
        "timestamp": float(lidar_message.get("timestamp", -1.0)),
        "objects_3d": objects_3d.to_payload(),
    }


def build_lanes_3d_frame_message(
    *,
    camera_message: Mapping[str, Any],
    lanes_3d: Lanes3DPrediction,
) -> Dict[str, Any]:
    return {
        "schema": "lanes_3d_frame",
        "frame": int(camera_message.get("frame", -1)),
        "timestamp": float(camera_message.get("timestamp", -1.0)),
        "lanes_3d": lanes_3d.to_payload(),
    }


def parse_objects_3d_frame_message(message: Mapping[str, Any]) -> Dict[str, Any]:
    payload = message.get("objects_3d", {})
    if not isinstance(payload, Mapping):
        payload = {}

    objects_3d = Objects3DPrediction.from_payload(payload)

    return {
        "schema": str(message.get("schema", "objects_3d_frame")),
        "frame": int(message.get("frame", -1)),
        "timestamp": float(message.get("timestamp", -1.0)),
        "objects_3d": objects_3d,
    }


def parse_lanes_3d_frame_message(message: Mapping[str, Any]) -> Dict[str, Any]:
    payload = message.get("lanes_3d", {})
    if not isinstance(payload, Mapping):
        payload = {}

    lanes_3d = Lanes3DPrediction.from_payload(payload)

    return {
        "schema": str(message.get("schema", "lanes_3d_frame")),
        "frame": int(message.get("frame", -1)),
        "timestamp": float(message.get("timestamp", -1.0)),
        "lanes_3d": lanes_3d,
    }
