import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

import carla
from src.carla.geometry.boxes import actor_to_gt_box
from src.common.config import CollectorConfig
from src.common.constants import NUSCENES_LIKE_CLASSES


def hero_to_dict(hero: carla.Actor) -> Dict:
    transform = hero.get_transform()
    bbox = hero.bounding_box
    return {
        "id": int(hero.id),
        "type_id": hero.type_id,
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


def lidar_to_dict(lidar_transform: carla.Transform) -> Dict:
    return {
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


def ego_bbox_to_gt_box(hero: carla.Actor, lidar_transform: carla.Transform) -> np.ndarray:
    return actor_to_gt_box(hero, lidar_transform)


def save_frame(
    output_root: Path,
    frame_index: int,
    frame_id: int,
    timestamp: float,
    world: carla.World,
    hero: carla.Actor,
    lidar_transform: carla.Transform,
    points: np.ndarray,
    objects: List[Dict],
    class_counts: Dict[str, int],
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    gt_ids: List[int],
    gt_type_ids: List[str],
    config: CollectorConfig,
) -> Path:
    frame_dir = output_root / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    np.save(frame_dir / "points.npy", points)
    np.save(frame_dir / "gt_boxes.npy", gt_boxes)
    np.save(frame_dir / "gt_names.npy", gt_names)
    np.save(frame_dir / "ego_box.npy", ego_bbox_to_gt_box(hero, lidar_transform))

    meta = {
        "frame": int(frame_id),
        "timestamp": float(timestamp),
        "map": world.get_map().name,
        "max_range": float(config.max_range),
        "ego_bbox_padding": float(config.ego_bbox_padding),
        "class_schema": "cityobject(car/truck/bus/motorcycle/bicycle) + actor(motorcycle/bicycle/pedestrian)",
        "classes": NUSCENES_LIKE_CLASSES,
        "num_points": int(points.shape[0]),
        "num_objects": int(len(objects)),
        "class_counts": class_counts,
        "hero": hero_to_dict(hero),
        "lidar_sensor": lidar_to_dict(lidar_transform),
        "ego_vehicle_box": ego_bbox_to_gt_box(hero, lidar_transform).tolist(),
        "objects": objects,
        "gt_ids": gt_ids,
        "gt_type_ids": gt_type_ids,
        "gt_names": gt_names.tolist(),
        "saved_at": datetime.now().isoformat(),
    }

    with open(frame_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return frame_dir
