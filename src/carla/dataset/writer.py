import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

import carla
from src.carla.geometry.boxes import actor_to_gt_box
from src.common.config import CameraConfig, CollectorConfig, LaneAnnotationsConfig, LidarConfig
from src.common.constants import NUSCENES_LIKE_CLASSES
from src.common.typing_aliases import Float32Array, Float64Array, ImageArray, StrArray


def hero_to_dict(hero: carla.Actor) -> Dict[str, object]:
    """Serialize the hero vehicle state into the dataset metadata format."""
    transform = hero.get_transform()
    bbox = hero.bounding_box
    return {
        "id": int(hero.id),
        "type_id": hero.type_id,
        "transform": {
            "location": {
                "x": float(transform.location.x),
                "y": float(transform.location.y),
                "z": float(transform.location.z),
            },
            "rotation": {
                "pitch": float(transform.rotation.pitch),
                "yaw": float(transform.rotation.yaw),
                "roll": float(transform.rotation.roll),
            },
        },
        "bounding_box": {
            "location": {
                "x": float(bbox.location.x),
                "y": float(bbox.location.y),
                "z": float(bbox.location.z),
            },
            "extent": {
                "x": float(bbox.extent.x),
                "y": float(bbox.extent.y),
                "z": float(bbox.extent.z),
            },
            "rotation": {
                "pitch": float(bbox.rotation.pitch),
                "yaw": float(bbox.rotation.yaw),
                "roll": float(bbox.rotation.roll),
            },
        },
    }


def lidar_to_dict(lidar_transform: carla.Transform, config: LidarConfig) -> Dict[str, object]:
    """Serialize LiDAR sensor settings and transform metadata."""
    return {
        "max_range": float(config.max_range),
        "channels": int(config.channels),
        "points_per_second": int(config.points_per_second),
        "upper_fov": float(config.upper_fov),
        "lower_fov": float(config.lower_fov),
        "location": {
            "x": float(lidar_transform.location.x),
            "y": float(lidar_transform.location.y),
            "z": float(lidar_transform.location.z),
        },
        "rotation": {
            "pitch": float(lidar_transform.rotation.pitch),
            "yaw": float(lidar_transform.rotation.yaw),
            "roll": float(lidar_transform.rotation.roll),
        },
    }


def camera_to_dict(camera_transform: carla.Transform, config: CameraConfig) -> Dict[str, object]:
    """Serialize camera settings and transform metadata."""
    return {
        "width": int(config.width),
        "height": int(config.height),
        "fov": float(config.fov),
        "mount": {
            "x": float(config.x),
            "y": float(config.y),
            "z": float(config.z),
            "pitch": float(config.pitch),
            "yaw": float(config.yaw),
            "roll": float(config.roll),
        },
        "location": {
            "x": float(camera_transform.location.x),
            "y": float(camera_transform.location.y),
            "z": float(camera_transform.location.z),
        },
        "rotation": {
            "pitch": float(camera_transform.rotation.pitch),
            "yaw": float(camera_transform.rotation.yaw),
            "roll": float(camera_transform.rotation.roll),
        },
    }


def lane_annotations_to_dict(config: LaneAnnotationsConfig) -> Dict[str, object]:
    """Serialize lane-annotation collection settings into metadata."""
    return {
        "distance_m": float(config.distance_m),
        "step_m": float(config.step_m),
        "max_side_lanes": int(config.max_side_lanes),
        "projection_margin_px": float(config.projection_margin_px),
        "dedupe_distance_px": float(config.dedupe_distance_px),
        "min_segment_points": int(config.min_segment_points),
        "min_projected_points": int(config.min_projected_points),
        "min_length_px": float(config.min_length_px),
        "min_length_m": float(config.min_length_m),
        "extend_to_bottom_threshold_px": float(config.extend_to_bottom_threshold_px),
    }


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write a JSON payload to disk with UTF-8 encoding."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_multimodal_frame(
    output_root: Path,
    frame_index: int,
    sim_frame: int,
    timestamp: float,
    world: carla.World,
    hero: carla.Actor,
    lidar_transform: carla.Transform,
    camera_transform: carla.Transform,
    points: Float64Array,
    image_bgr: ImageArray,
    lanes: List[Dict[str, object]],
    objects: List[Dict[str, object]],
    class_counts: Dict[str, int],
    gt_boxes: Float32Array,
    gt_names: StrArray,
    config: CollectorConfig,
) -> Path:
    """Write one synchronized multimodal CARLA frame to the raw dataset layout."""
    frame_dir = output_root / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    points_path = frame_dir / "points.npy"
    gt_boxes_path = frame_dir / "gt_boxes.npy"
    gt_names_path = frame_dir / "gt_names.npy"
    ego_box_path = frame_dir / "ego_box.npy"
    rgb_path = frame_dir / "front_rgb.png"
    objects_path = frame_dir / "objects.json"
    lanes_path = frame_dir / "lanes.json"
    meta_path = frame_dir / "meta.json"

    np.save(points_path, points)
    np.save(gt_boxes_path, gt_boxes)
    np.save(gt_names_path, gt_names)
    np.save(ego_box_path, actor_to_gt_box(hero, lidar_transform))
    cv2.imwrite(str(rgb_path), image_bgr)

    _write_json(objects_path, {"objects": objects})

    _write_json(lanes_path, {"lanes": lanes})

    meta = {
        "frame_index": int(frame_index),
        "sim_frame": int(sim_frame),
        "saved_at": datetime.now().isoformat(),
        "timestamp": float(timestamp),
        "map": world.get_map().name,
        "hero": hero_to_dict(hero),
        "lidar": lidar_to_dict(lidar_transform, config.lidar),
        "front_camera": camera_to_dict(camera_transform, config.camera_front),
        "lane_annotations": lane_annotations_to_dict(config.lane_annotations),
        "num_points": int(points.shape[0]),
        "classes": NUSCENES_LIKE_CLASSES,
        "num_objects": int(len(objects)),
        "class_counts": class_counts,
        "num_lanes": int(len(lanes)),
    }
    _write_json(meta_path, meta)

    return frame_dir
