#!/usr/bin/env python3

import argparse
import os
import queue
import time

import carla
from src.carla.actors.classify import find_hero_vehicle
from src.carla.dataset.writer import save_frame
from src.carla.gt.collector import collect_all_gt, count_by_class
from src.carla.lidar.frame_buffer import LidarFrameBuffer
from src.carla.lidar.processing import preprocess_lidar_points
from src.carla.lidar.sensor import configure_lidar_blueprint
from src.common.constants import NUSCENES_LIKE_CLASSES
from src.common.runtime_config import build_collector_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CARLA detector dataset (ray_cast LiDAR)")
    parser.add_argument("-o", "--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("-n", "--num-frames", type=int, default=100, help="How many frames to save")
    parser.add_argument("--every-nth", type=int, default=5, help="Save every N-th LiDAR frame")
    args = parser.parse_args()

    config = build_collector_config(
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        every_nth=args.every_nth,
    )

    client = carla.Client(config.carla.host, config.carla.port)
    client.set_timeout(5.0)
    world = client.get_world()

    settings = world.get_settings()
    if not settings.synchronous_mode:
        raise RuntimeError("Run manual_control.py with the --sync flag")

    hero = find_hero_vehicle(world)
    if hero is None:
        raise RuntimeError("Hero vehicle not found")

    print(f"[info] hero id={hero.id}, type={hero.type_id}")
    print(f"[info] output dir: {config.output_dir}")
    print(f"[info] num_frames: {config.num_frames}")
    print(f"[info] every_nth: {config.every_nth}")
    print(f"[info] max_range: {config.max_range} m")
    print(f"[info] ego_bbox_padding: {config.ego_bbox_padding} m")
    print("[info] GT source: cityobject(car/truck/bus/motorcycle/bicycle) + actors(motorcycle/bicycle/pedestrian)")

    lidar_bp = configure_lidar_blueprint(world, config=config.lidar, fixed_delta_seconds=settings.fixed_delta_seconds)

    lidar_transform_relative = carla.Transform(carla.Location(x=-0.5, z=1.8))
    lidar = None

    os.makedirs(config.output_dir, exist_ok=True)

    try:
        lidar = world.spawn_actor(lidar_bp, lidar_transform_relative, attach_to=hero)
        frame_buffer = LidarFrameBuffer(hero=hero, lidar=lidar)
        lidar.listen(frame_buffer.callback)

        print("[info] waiting for the first LiDAR frame...")
        start = time.time()

        while True:
            snapshot = world.wait_for_tick()
            expected_frame = int(snapshot.frame) if hasattr(snapshot, "frame") else int(snapshot)
            try:
                first_raw_points, first_frame, _, _ = frame_buffer.get_frame(expected_frame)
                first_points = preprocess_lidar_points(
                    points=first_raw_points,
                    hero=hero,
                    lidar=lidar,
                    ego_bbox_padding=config.ego_bbox_padding,
                )
                print(f"[info] first frame: frame={first_frame}, points={first_points.shape[0]}")
                break
            except queue.Empty:
                if time.time() - start > 10.0:
                    raise RuntimeError("Timeout: no LiDAR frame received")

        saved_count = 0
        last_saved_frame = None

        while saved_count < config.num_frames:
            snapshot = world.wait_for_tick()
            expected_frame = int(snapshot.frame) if hasattr(snapshot, "frame") else int(snapshot)

            try:
                raw_points, frame_snapshot, timestamp_snapshot, lidar_transform_snapshot = frame_buffer.get_frame(
                    expected_frame
                )
            except queue.Empty:
                continue

            points_snapshot = preprocess_lidar_points(
                points=raw_points,
                hero=hero,
                lidar=lidar,
                ego_bbox_padding=config.ego_bbox_padding,
            )

            if last_saved_frame is not None and frame_snapshot == last_saved_frame:
                continue

            if frame_snapshot % config.every_nth != 0:
                continue

            objects, gt_boxes, gt_names, gt_ids, gt_type_ids = collect_all_gt(
                world=world,
                hero=hero,
                lidar_transform=lidar_transform_snapshot,
                max_range=config.max_range,
            )

            class_counts = count_by_class(objects)

            frame_dir = save_frame(
                output_root=config.output_dir,
                frame_index=saved_count,
                frame_id=frame_snapshot,
                timestamp=timestamp_snapshot,
                world=world,
                hero=hero,
                lidar_transform=lidar_transform_snapshot,
                points=points_snapshot,
                objects=objects,
                class_counts=class_counts,
                gt_boxes=gt_boxes,
                gt_names=gt_names,
                gt_ids=gt_ids,
                gt_type_ids=gt_type_ids,
                config=config,
            )

            print(
                f"[saved {saved_count + 1}/{config.num_frames}] "
                f"{frame_dir} | frame={frame_snapshot} | points={points_snapshot.shape[0]} | "
                + " | ".join(f"{k}={class_counts.get(k, 0)}" for k in NUSCENES_LIKE_CLASSES)
            )

            last_saved_frame = frame_snapshot
            saved_count += 1

        print("[done] detector dataset collection finished")

    finally:
        if lidar is not None:
            try:
                lidar.stop()
            except RuntimeError:
                pass
            lidar.destroy()


if __name__ == "__main__":
    main()
