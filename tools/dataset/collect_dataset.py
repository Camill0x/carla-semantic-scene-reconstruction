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
from src.common.config import CollectorConfig
from src.common.constants import NUSCENES_LIKE_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CARLA detector dataset (ray_cast LiDAR)")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--output-dir", default="run_01", help="Output dataset directory")
    parser.add_argument("--num-frames", type=int, default=100, help="How many frames to save")
    parser.add_argument("--every-nth", type=int, default=5, help="Save every N-th LiDAR frame")
    parser.add_argument("--range", dest="lidar_range", type=float, default=50.0, help="LiDAR/object range in meters")
    parser.add_argument("--channels", type=int, default=32, help="LiDAR channels")
    parser.add_argument("--points-per-second", type=int, default=700000, help="LiDAR points per second")
    parser.add_argument("--upper-fov", type=float, default=10.0, help="LiDAR upper FOV")
    parser.add_argument("--lower-fov", type=float, default=-30.0, help="LiDAR lower FOV")
    parser.add_argument(
        "--ego-bbox-padding",
        type=float,
        default=0.15,
        help="Extra padding [m] used when removing LiDAR points that fall inside ego vehicle bbox",
    )
    args = parser.parse_args()

    if args.lidar_range <= 0.0:
        raise ValueError("--range must be > 0")
    if args.ego_bbox_padding < 0.0:
        raise ValueError("--ego-bbox-padding must be >= 0")
    if args.every_nth <= 0:
        raise ValueError("--every-nth must be >= 1")

    config = CollectorConfig(
        max_range=float(args.lidar_range),
        ego_bbox_padding=float(args.ego_bbox_padding),
        output_dir=args.output_dir,
        num_frames=int(args.num_frames),
        every_nth=int(args.every_nth),
    )

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    settings = world.get_settings()
    if not settings.synchronous_mode:
        raise RuntimeError("Uruchom manual_control.py z flagą --sync")

    hero = find_hero_vehicle(world)
    if hero is None:
        raise RuntimeError("Nie znaleziono pojazdu hero")

    print(f"[info] hero id={hero.id}, type={hero.type_id}")
    print(f"[info] output dir: {config.output_dir}")
    print(f"[info] num_frames: {config.num_frames}")
    print(f"[info] every_nth: {config.every_nth}")
    print(f"[info] max_range: {config.max_range} m")
    print(f"[info] ego_bbox_padding: {config.ego_bbox_padding} m")
    print("[info] GT source: cityobject(car/truck/bus/motorcycle/bicycle) + actors(motorcycle/bicycle/pedestrian)")

    bp_lib = world.get_blueprint_library()
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("upper_fov", str(args.upper_fov))
    lidar_bp.set_attribute("lower_fov", str(args.lower_fov))
    lidar_bp.set_attribute("channels", str(args.channels))
    lidar_bp.set_attribute("range", str(config.max_range))
    lidar_bp.set_attribute("points_per_second", str(args.points_per_second))

    delta = settings.fixed_delta_seconds if settings.fixed_delta_seconds else 0.05
    lidar_bp.set_attribute("rotation_frequency", str(1.0 / delta))
    lidar_bp.set_attribute("noise_stddev", "0.0")
    lidar_bp.set_attribute("dropoff_general_rate", "0.0")
    lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
    lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")

    lidar_transform_relative = carla.Transform(carla.Location(x=-0.5, z=1.8))
    lidar = None

    os.makedirs(config.output_dir, exist_ok=True)

    try:
        lidar = world.spawn_actor(lidar_bp, lidar_transform_relative, attach_to=hero)
        frame_buffer = LidarFrameBuffer(hero=hero, lidar=lidar)
        lidar.listen(frame_buffer.callback)

        print("[info] czekam na pierwszą klatkę lidaru...")
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
                print(f"[info] pierwsza klatka: frame={first_frame}, points={first_points.shape[0]}")
                break
            except queue.Empty:
                if time.time() - start > 10.0:
                    raise RuntimeError("Timeout: nie przyszła żadna klatka lidaru")

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
