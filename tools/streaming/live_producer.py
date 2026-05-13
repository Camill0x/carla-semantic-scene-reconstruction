#!/usr/bin/env python3

import argparse
import queue
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

import carla
from src.carla.actors.classify import find_hero_vehicle
from src.carla.camera.frame_buffer import CameraFrameBuffer
from src.carla.camera.sensor import configure_front_camera_blueprint, front_camera_transform
from src.carla.geometry.boxes import actor_to_gt_box
from src.carla.lidar.frame_buffer import LidarFrameBuffer
from src.carla.lidar.processing import preprocess_lidar_points
from src.carla.lidar.sensor import configure_lidar_blueprint
from src.common.cli_logging import configure_logging
from src.common.runtime_config import build_streaming_producer_config
from src.shared_memory.buffers import SharedArrayPool, SharedMessageBuffer
from src.shared_memory.names import SharedMemoryNames
from src.streaming.messages import build_frame_snapshot_message, build_state_frame_message


@dataclass(frozen=True)
class LiveProducerArgs:
    every_nth: int
    verbose: bool


def parse_args() -> LiveProducerArgs:
    parser = argparse.ArgumentParser(description="CARLA live producer")
    parser.add_argument("--every-nth", type=int, default=1, help="Publish every N-th CARLA frame")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame logs")
    parsed = parser.parse_args()
    return LiveProducerArgs(
        every_nth=int(parsed.every_nth),
        verbose=bool(parsed.verbose),
    )


def transform_to_dict(transform: carla.Transform) -> Dict[str, object]:
    return {
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
    }


def should_process_frame(frame: int, last_processed_frame: Optional[int], every_nth: int) -> bool:
    return last_processed_frame is None or frame - last_processed_frame >= every_nth


def main() -> None:
    args = parse_args()
    logger = configure_logging("tools.streaming.live_producer", verbose=args.verbose)
    config = build_streaming_producer_config(every_nth=args.every_nth)
    names = SharedMemoryNames(prefix=config.common.prefix)

    camera_slot_capacity = (
        int(config.camera_front.width) * int(config.camera_front.height) * 3 * np.dtype(np.uint8).itemsize
    )
    camera_pool = SharedArrayPool(
        prefix=names.camera_prefix,
        slot_capacity_bytes=camera_slot_capacity,
        num_slots=config.sensor_slots,
    )
    lidar_pool = SharedArrayPool(
        prefix=names.lidar_prefix,
        slot_capacity_bytes=config.lidar_slot_capacity_bytes,
        num_slots=config.sensor_slots,
    )
    frame_buffer = SharedMessageBuffer(
        name=names.frame_buffer,
        size_bytes=config.common.frame_buffer_size_bytes,
        create=True,
    )
    objects_buffer = SharedMessageBuffer(
        name=names.objects_buffer,
        size_bytes=config.common.objects_buffer_size_bytes,
        create=True,
    )
    lanes_buffer = SharedMessageBuffer(
        name=names.lanes_buffer,
        size_bytes=config.common.lanes_buffer_size_bytes,
        create=True,
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

    logger.info("hero id=%s, type=%s", hero.id, hero.type_id)
    logger.info("frame buffer: %s", names.frame_buffer)
    logger.info("camera prefix: %s", names.camera_prefix)
    logger.info("lidar prefix: %s", names.lidar_prefix)
    logger.info("sensor_slots: %s", config.sensor_slots)

    lidar_bp = configure_lidar_blueprint(world, config=config.lidar, fixed_delta_seconds=settings.fixed_delta_seconds)
    lidar_transform_relative = carla.Transform(carla.Location(x=-0.5, z=1.8))
    camera_bp = configure_front_camera_blueprint(
        world,
        config=config.camera_front,
        fixed_delta_seconds=settings.fixed_delta_seconds,
    )

    lidar = None
    camera = None

    try:
        lidar = world.spawn_actor(lidar_bp, lidar_transform_relative, attach_to=hero)
        camera = world.spawn_actor(
            camera_bp,
            front_camera_transform(config.camera_front),
            attach_to=hero,
            attachment_type=carla.AttachmentType.Rigid,
        )

        lidar_frame_buffer = LidarFrameBuffer(hero=hero, lidar=lidar)
        camera_frame_buffer = CameraFrameBuffer(camera=camera)
        lidar.listen(lidar_frame_buffer.callback)
        camera.listen(camera_frame_buffer.callback)

        last_published_frame = None
        published_count = 0

        while True:
            snapshot = world.wait_for_tick()
            expected_frame = int(snapshot.frame) if hasattr(snapshot, "frame") else int(snapshot)

            if not should_process_frame(expected_frame, last_published_frame, config.every_nth):
                next_frame = (
                    config.every_nth if last_published_frame is None else last_published_frame + config.every_nth
                )
                lidar_frame_buffer.discard_before(next_frame)
                camera_frame_buffer.discard_before(next_frame)
                continue

            try:
                lidar_data = lidar_frame_buffer.get_frame(
                    expected_frame=expected_frame,
                    timeout=2.0,
                    allow_future=True,
                )
                camera_data = camera_frame_buffer.get_frame(
                    expected_frame=expected_frame,
                    timeout=2.0,
                    allow_future=True,
                )

                while lidar_data[1] != camera_data[1]:
                    target_frame = max(lidar_data[1], camera_data[1])
                    if lidar_data[1] < target_frame:
                        lidar_data = lidar_frame_buffer.get_frame(
                            expected_frame=target_frame,
                            timeout=2.0,
                            allow_future=True,
                        )
                    if camera_data[1] < target_frame:
                        camera_data = camera_frame_buffer.get_frame(
                            expected_frame=target_frame,
                            timeout=2.0,
                            allow_future=True,
                        )
            except queue.Empty:
                if last_published_frame is None:
                    raise RuntimeError("Timeout: no input data received")
                continue

            raw_points, frame_snapshot, lidar_timestamp, lidar_transform_snapshot = lidar_data
            camera_image_bgr_snapshot, _camera_frame, camera_timestamp, camera_transform_snapshot = camera_data
            timestamp_snapshot = max(lidar_timestamp, camera_timestamp)

            if last_published_frame is not None and frame_snapshot == last_published_frame:
                continue

            if not should_process_frame(frame_snapshot, last_published_frame, config.every_nth):
                continue

            points_snapshot = preprocess_lidar_points(
                points=np.asarray(raw_points, dtype=np.float64),
                hero=hero,
                lidar=lidar,
            ).astype(np.float32)
            camera_image_bgr_snapshot = np.asarray(camera_image_bgr_snapshot, dtype=np.uint8)
            ego_box = actor_to_gt_box(hero, lidar_transform_snapshot).astype(np.float32)

            slot_index = int(frame_snapshot) % config.sensor_slots
            camera_descriptor = camera_pool.write(camera_image_bgr_snapshot, slot_index=slot_index)
            lidar_descriptor = lidar_pool.write(points_snapshot, slot_index=slot_index)

            state_message = build_state_frame_message(
                frame=int(frame_snapshot),
                timestamp=float(timestamp_snapshot),
                ego_box=ego_box,
                lidar_metadata={
                    "transform": transform_to_dict(lidar_transform_snapshot),
                    "ground_z": -float(lidar_transform_relative.location.z),
                },
                camera_front_metadata={
                    "fov": float(config.camera_front.fov),
                    "transform": transform_to_dict(camera_transform_snapshot),
                },
            )

            frame_buffer.write(
                build_frame_snapshot_message(
                    frame=int(frame_snapshot),
                    timestamp=float(timestamp_snapshot),
                    published_at=time.monotonic_ns(),
                    camera_descriptor=camera_descriptor,
                    lidar_descriptor=lidar_descriptor,
                    state_message=state_message,
                )
            )

            last_published_frame = frame_snapshot
            published_count += 1
            logger.debug("published_frame=%s", frame_snapshot)

    finally:
        if lidar is not None:
            try:
                lidar.stop()
            except RuntimeError:
                pass
            lidar.destroy()
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass
            camera.destroy()
        frame_buffer.close()
        objects_buffer.close()
        lanes_buffer.close()
        camera_pool.close()
        lidar_pool.close()


if __name__ == "__main__":
    main()
