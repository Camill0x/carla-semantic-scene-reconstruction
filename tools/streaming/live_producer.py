#!/usr/bin/env python3

import argparse
import queue
import time

import numpy as np
import zmq

import carla
from src.carla.actors.classify import find_hero_vehicle
from src.carla.gt.collector import collect_all_gt, count_by_class
from src.carla.lidar.frame_buffer import LidarFrameBuffer
from src.carla.lidar.processing import preprocess_lidar_points
from src.common.constants import NUSCENES_LIKE_CLASSES
from src.streaming.messages import build_lidar_message


def main() -> None:
    parser = argparse.ArgumentParser(description="CARLA live LiDAR producer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--zmq-bind", default="tcp://*:5555")
    parser.add_argument("--every-nth", type=int, default=1)
    parser.add_argument("--range", dest="lidar_range", type=float, default=50.0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--points-per-second", type=int, default=700000)
    parser.add_argument("--upper-fov", type=float, default=10.0)
    parser.add_argument("--lower-fov", type=float, default=-30.0)
    parser.add_argument("--ego-bbox-padding", type=float, default=0.15)
    parser.add_argument("--with-gt", action="store_true")
    args = parser.parse_args()

    if args.lidar_range <= 0.0:
        raise ValueError("--range must be > 0")
    if args.every_nth <= 0:
        raise ValueError("--every-nth must be >= 1")
    if args.ego_bbox_padding < 0.0:
        raise ValueError("--ego-bbox-padding must be >= 0")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind(args.zmq_bind)

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    settings = world.get_settings()
    if not settings.synchronous_mode:
        raise RuntimeError("Run manual_control.py with the --sync flag")

    hero = find_hero_vehicle(world)
    if hero is None:
        raise RuntimeError("Hero vehicle not found")

    print(f"[info] hero id={hero.id}, type={hero.type_id}")
    print(f"[info] ZMQ bind: {args.zmq_bind}")
    print(f"[info] every_nth: {args.every_nth}")
    print(f"[info] max_range: {args.lidar_range}")
    print(f"[info] with_gt: {args.with_gt}")

    bp_lib = world.get_blueprint_library()
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("upper_fov", str(args.upper_fov))
    lidar_bp.set_attribute("lower_fov", str(args.lower_fov))
    lidar_bp.set_attribute("channels", str(args.channels))
    lidar_bp.set_attribute("range", str(args.lidar_range))
    lidar_bp.set_attribute("points_per_second", str(args.points_per_second))

    delta = settings.fixed_delta_seconds if settings.fixed_delta_seconds else 0.05
    lidar_bp.set_attribute("rotation_frequency", str(1.0 / delta))
    lidar_bp.set_attribute("noise_stddev", "0.0")
    lidar_bp.set_attribute("dropoff_general_rate", "0.0")
    lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
    lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")

    lidar_transform_relative = carla.Transform(carla.Location(x=-0.5, z=1.8))
    lidar = None

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
                raw_points, first_frame, _, _ = frame_buffer.get_frame(expected_frame)
                first_points = preprocess_lidar_points(
                    points=raw_points,
                    hero=hero,
                    lidar=lidar,
                    ego_bbox_padding=float(args.ego_bbox_padding),
                )
                print(f"[info] first frame: frame={first_frame}, points={first_points.shape[0]}")
                break
            except queue.Empty:
                if time.time() - start > 10.0:
                    raise RuntimeError("Timeout: no LiDAR frame received")

        # time.sleep(0.3)

        last_published_frame = None
        t_last_log = time.time()
        published_count = 0

        while True:
            snapshot = world.wait_for_tick()
            expected_frame = int(snapshot.frame) if hasattr(snapshot, "frame") else int(snapshot)

            try:
                raw_points, frame_snapshot, timestamp_snapshot, lidar_transform_snapshot = frame_buffer.get_frame(
                    expected_frame
                )
            except queue.Empty:
                continue

            if last_published_frame is not None and frame_snapshot == last_published_frame:
                continue

            if frame_snapshot % args.every_nth != 0:
                continue

            points_snapshot = preprocess_lidar_points(
                points=raw_points,
                hero=hero,
                lidar=lidar,
                ego_bbox_padding=float(args.ego_bbox_padding),
            )

            gt_payload = None

            if args.with_gt:
                objects, gt_boxes, gt_names, gt_ids, gt_type_ids = collect_all_gt(
                    world=world,
                    hero=hero,
                    lidar_transform=lidar_transform_snapshot,
                    max_range=float(args.lidar_range),
                )

                gt_payload = {
                    "num_objects": int(len(objects)),
                    "class_counts": count_by_class(objects),
                    "gt_ids": gt_ids,
                    "gt_type_ids": gt_type_ids,
                    "gt_names": gt_names.tolist(),
                    "objects": objects,
                    "gt_boxes": gt_boxes.astype(np.float32),
                }

            message = build_lidar_message(
                frame=int(frame_snapshot),
                timestamp=float(timestamp_snapshot),
                max_range=float(args.lidar_range),
                classes=NUSCENES_LIKE_CLASSES,
                hero_id=int(hero.id),
                hero_type_id=hero.type_id,
                points=points_snapshot.astype(np.float32),
                gt_payload=gt_payload,
            )

            socket.send_pyobj(message)

            last_published_frame = frame_snapshot
            published_count += 1

            now = time.time()
            if now - t_last_log >= 1.0:
                print(
                    f"[producer] published={published_count} | frame={frame_snapshot} | points={points_snapshot.shape[0]}"
                )
                t_last_log = now

    finally:
        if lidar is not None:
            try:
                lidar.stop()
            except RuntimeError:
                pass
            lidar.destroy()
        socket.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
