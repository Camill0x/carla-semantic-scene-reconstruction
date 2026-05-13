from typing import Dict, List, Sequence, Tuple

import numpy as np

import carla
from src.carla.actors.classify import classify_vehicle_actor
from src.carla.geometry.boxes import actor_matches_level_bbox, actor_to_gt_box, level_bbox_to_gt_box
from src.carla.geometry.transforms import distance_between_locations
from src.common.typing_aliases import Float32Array, ObjectDict, StrArray


def actor_to_object_dict(actor: carla.Actor, cls_name: str) -> ObjectDict:
    """Serialize a dynamic CARLA actor into the object-annotation format."""
    transform = actor.get_transform()
    bbox = actor.bounding_box

    return {
        "source": "actor",
        "id": int(actor.id),
        "type_id": actor.type_id,
        "class_name": cls_name,
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


def level_bbox_to_object_dict(
    bbox: carla.BoundingBox,
    cls_name: str,
    static_id: int,
    type_id: str,
) -> ObjectDict:
    """Serialize a static level bounding box into the object-annotation format."""
    return {
        "source": "static_level_bbox",
        "id": int(static_id),
        "type_id": type_id,
        "class_name": cls_name,
        "transform": {
            "location": {
                "x": float(bbox.location.x),
                "y": float(bbox.location.y),
                "z": float(bbox.location.z),
            },
            "rotation": {
                "pitch": float(bbox.rotation.pitch),
                "yaw": float(bbox.rotation.yaw),
                "roll": float(bbox.rotation.roll),
            },
        },
        "bounding_box": {
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "extent": {
                "x": float(bbox.extent.x),
                "y": float(bbox.extent.y),
                "z": float(bbox.extent.z),
            },
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        },
    }


def collect_city_object_gt(
    world: carla.World,
    hero: carla.Actor,
    lidar_transform: carla.Transform,
    max_range: float,
) -> Tuple[List[ObjectDict], List[Float32Array], List[str]]:
    """Collect static city-object annotations around the ego vehicle."""
    hero_location = hero.get_transform().location

    objects: List[ObjectDict] = []
    gt_boxes: List[Float32Array] = []
    gt_names: List[str] = []

    label_map = [
        (carla.CityObjectLabel.Car, "car"),
        (carla.CityObjectLabel.Truck, "truck"),
        (carla.CityObjectLabel.Bus, "bus"),
        (carla.CityObjectLabel.Motorcycle, "motorcycle"),
        (carla.CityObjectLabel.Bicycle, "bicycle"),
    ]

    next_static_id = -1

    for label, cls_name in label_map:
        for bbox in world.get_level_bbs(label):
            if distance_between_locations(bbox.location, hero_location) > max_range:
                continue

            if cls_name in {"motorcycle", "bicycle"} and float(bbox.extent.y) == 0.0:
                continue

            if actor_matches_level_bbox(actor=hero, level_bbox=bbox):
                continue

            type_id = f"city_object.{cls_name}"
            gt_box = level_bbox_to_gt_box(bbox, lidar_transform)

            obj = level_bbox_to_object_dict(
                bbox=bbox,
                cls_name=cls_name,
                static_id=next_static_id,
                type_id=type_id,
            )

            objects.append(obj)
            gt_boxes.append(gt_box)
            gt_names.append(cls_name)
            next_static_id -= 1

    return objects, gt_boxes, gt_names


def collect_actor_gt(
    world: carla.World,
    hero: carla.Actor,
    lidar_transform: carla.Transform,
    max_range: float,
) -> Tuple[List[ObjectDict], List[Float32Array], List[str]]:
    """Collect dynamic actor annotations around the ego vehicle."""
    hero_location = hero.get_transform().location

    objects: List[ObjectDict] = []
    gt_boxes: List[Float32Array] = []
    gt_names: List[str] = []

    for actor in world.get_actors():
        if actor.id == hero.id:
            continue

        cls_name = None
        if actor.type_id.startswith("walker.pedestrian."):
            cls_name = "pedestrian"
        elif actor.type_id.startswith("vehicle."):
            vehicle_cls = classify_vehicle_actor(actor)
            if vehicle_cls in {"motorcycle", "bicycle"}:
                cls_name = vehicle_cls

        if cls_name is None:
            continue

        if distance_between_locations(actor.get_transform().location, hero_location) > max_range:
            continue

        obj = actor_to_object_dict(actor, cls_name)
        gt_box = actor_to_gt_box(actor, lidar_transform)

        objects.append(obj)
        gt_boxes.append(gt_box)
        gt_names.append(cls_name)

    return objects, gt_boxes, gt_names


def collect_gt(
    world: carla.World,
    hero: carla.Actor,
    lidar_transform: carla.Transform,
    max_range: float,
) -> Tuple[List[ObjectDict], Float32Array, StrArray]:
    """Collect and merge static and dynamic ground-truth object annotations."""
    city_objects, city_boxes, city_names = collect_city_object_gt(
        world=world,
        hero=hero,
        lidar_transform=lidar_transform,
        max_range=max_range,
    )

    actor_objects, actor_boxes, actor_names = collect_actor_gt(
        world=world,
        hero=hero,
        lidar_transform=lidar_transform,
        max_range=max_range,
    )

    objects = city_objects + actor_objects
    gt_boxes = city_boxes + actor_boxes
    gt_names = city_names + actor_names

    if gt_boxes:
        gt_boxes_array = np.stack(gt_boxes, axis=0).astype(np.float32)
        gt_names_array = np.array(gt_names)
    else:
        gt_boxes_array = np.zeros((0, 7), dtype=np.float32)
        gt_names_array = np.array([], dtype="<U16")

    return objects, gt_boxes_array, gt_names_array


def count_by_class(objects: Sequence[ObjectDict]) -> Dict[str, int]:
    """Count annotated objects by class name."""
    counts: Dict[str, int] = {}
    for obj in objects:
        cls_name = str(obj["class_name"])
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts
