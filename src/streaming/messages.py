from typing import Any, Mapping, Optional

import numpy as np

from src.common.typing_aliases import Float32Array, ImageArray, JsonDict
from src.lanedet.prediction import Lanes3DPrediction
from src.openpcdet.prediction import Objects3DPrediction
from src.shared_memory.buffers import SharedArrayDescriptor


def build_lidar_frame_message(
    *,
    frame: int,
    timestamp: float,
    points: Float32Array,
) -> JsonDict:
    """Build the shared payload for one LiDAR frame."""
    points_array = np.asarray(points, dtype=np.float32)
    return {
        "schema": "lidar_frame",
        "frame": int(frame),
        "timestamp": float(timestamp),
        "lidar": {
            "points": points_array,
        },
    }


def parse_lidar_frame_message(message: Mapping[str, Any]) -> JsonDict:
    """Validate and normalize a serialized LiDAR frame message."""
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
    camera_front_image: ImageArray,
) -> JsonDict:
    """Build the shared payload for one camera frame."""
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


def parse_camera_frame_message(message: Mapping[str, Any]) -> JsonDict:
    """Validate and normalize a serialized camera frame message."""
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
    ego_box: Float32Array,
    lidar_metadata: Mapping[str, Any],
    camera_front_metadata: Mapping[str, Any],
) -> JsonDict:
    """Build the shared payload for one ego-state frame."""
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


def parse_state_frame_message(message: Mapping[str, Any]) -> JsonDict:
    """Validate and normalize a serialized ego-state frame message."""
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
) -> JsonDict:
    """Build the shared payload for one 3D object prediction frame."""
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
) -> JsonDict:
    """Build the shared payload for one 3D lane prediction frame."""
    return {
        "schema": "lanes_3d_frame",
        "frame": int(camera_message.get("frame", -1)),
        "timestamp": float(camera_message.get("timestamp", -1.0)),
        "lanes_3d": lanes_3d.to_payload(),
    }


def parse_objects_3d_frame_message(message: Mapping[str, Any]) -> JsonDict:
    """Validate and normalize a serialized 3D object prediction frame."""
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


def parse_lanes_3d_frame_message(message: Mapping[str, Any]) -> JsonDict:
    """Validate and normalize a serialized 3D lane prediction frame."""
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


def build_frame_snapshot_message(
    *,
    frame: int,
    timestamp: float,
    published_at: int,
    camera_descriptor: SharedArrayDescriptor,
    lidar_descriptor: SharedArrayDescriptor,
    state_message: Mapping[str, Any],
) -> JsonDict:
    """Build a combined frame snapshot payload for the producer output."""
    return {
        "frame": int(frame),
        "timestamp": float(timestamp),
        "published_at_monotonic_ns": int(published_at),
        "camera_front": {
            "height": int(camera_descriptor.shape[0]),
            "width": int(camera_descriptor.shape[1]),
            "shared_array": camera_descriptor.to_payload(),
        },
        "lidar": {
            "shared_array": lidar_descriptor.to_payload(),
        },
        "state": dict(state_message),
    }


def parse_frame_snapshot_message(message: Mapping[str, Any]) -> JsonDict:
    """Validate and normalize a combined frame snapshot payload."""
    camera_payload = message.get("camera_front")
    lidar_payload = message.get("lidar")
    state_payload = message.get("state")
    if not isinstance(camera_payload, Mapping):
        raise ValueError("Missing camera_front payload")
    if not isinstance(lidar_payload, Mapping):
        raise ValueError("Missing lidar payload")
    if not isinstance(state_payload, Mapping):
        raise ValueError("Missing state payload")

    camera_descriptor = SharedArrayDescriptor.from_payload(dict(camera_payload["shared_array"]))
    lidar_descriptor = SharedArrayDescriptor.from_payload(dict(lidar_payload["shared_array"]))
    return {
        "frame": int(message.get("frame", -1)),
        "timestamp": float(message.get("timestamp", -1.0)),
        "published_at_monotonic_ns": int(message.get("published_at_monotonic_ns", 0)),
        "camera_front": {
            "height": int(camera_payload.get("height", camera_descriptor.shape[0])),
            "width": int(camera_payload.get("width", camera_descriptor.shape[1])),
            "shared_array": camera_descriptor.to_payload(),
        },
        "lidar": {
            "shared_array": lidar_descriptor.to_payload(),
        },
        "state": parse_state_frame_message(state_payload),
    }


def build_scene_frame_message(
    *,
    frame_message: Mapping[str, Any],
    objects_message: Optional[Mapping[str, Any]],
    lanes_message: Optional[Mapping[str, Any]],
) -> JsonDict:
    """Build an aggregated live-scene payload from producer and detector messages."""
    parsed_frame = parse_frame_snapshot_message(frame_message)
    parsed_state = parsed_frame["state"]
    parsed_objects = None
    parsed_lanes = None

    if objects_message is not None:
        objects_payload = objects_message.get("objects_3d") if isinstance(objects_message, Mapping) else None
        if isinstance(objects_payload, Objects3DPrediction):
            parsed_objects = dict(objects_message)
        else:
            parsed_objects = parse_objects_3d_frame_message(objects_message)

    if lanes_message is not None:
        lanes_payload = lanes_message.get("lanes_3d") if isinstance(lanes_message, Mapping) else None
        if isinstance(lanes_payload, Lanes3DPrediction):
            parsed_lanes = dict(lanes_message)
        else:
            parsed_lanes = parse_lanes_3d_frame_message(lanes_message)

    return {
        "schema": "scene_frame",
        "frame": int(parsed_state["frame"]),
        "timestamp": float(parsed_state["timestamp"]),
        "published_at_monotonic_ns": int(parsed_frame.get("published_at_monotonic_ns", 0)),
        "ego": parsed_state["ego"],
        "objects_3d": (
            parsed_objects["objects_3d"].to_payload()
            if parsed_objects is not None
            else Objects3DPrediction.empty().to_payload()
        ),
        "lanes_3d": (
            parsed_lanes["lanes_3d"].to_payload()
            if parsed_lanes is not None
            else Lanes3DPrediction.empty().to_payload()
        ),
        "source_frames": {
            "state": int(parsed_state["frame"]),
            "objects_3d": int(parsed_objects["frame"]) if parsed_objects is not None else None,
            "lanes_3d": int(parsed_lanes["frame"]) if parsed_lanes is not None else None,
        },
    }


def parse_scene_frame_message(message: Mapping[str, Any]) -> JsonDict:
    """Validate and normalize an aggregated live-scene payload."""
    ego = message.get("ego")
    if not isinstance(ego, Mapping):
        ego = {}
    source_frames = message.get("source_frames")
    if not isinstance(source_frames, Mapping):
        source_frames = {}
    ego_box = np.asarray(ego.get("box", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    return {
        "schema": str(message.get("schema", "scene_frame")),
        "frame": int(message.get("frame", -1)),
        "timestamp": float(message.get("timestamp", -1.0)),
        "published_at_monotonic_ns": int(message.get("published_at_monotonic_ns", 0)),
        "ego": {"box": ego_box},
        "objects_3d": Objects3DPrediction.from_payload(message.get("objects_3d", {})),
        "lanes_3d": Lanes3DPrediction.from_payload(message.get("lanes_3d", {})),
        "source_frames": {
            "state": int(source_frames.get("state", message.get("frame", -1))),
            "objects_3d": int(source_frames["objects_3d"]) if source_frames.get("objects_3d") is not None else None,
            "lanes_3d": int(source_frames["lanes_3d"]) if source_frames.get("lanes_3d") is not None else None,
        },
    }
