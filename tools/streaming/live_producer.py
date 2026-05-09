#!/usr/bin/env python3

import argparse
import queue
from typing import Optional

import numpy as np
import zmq

import carla
from src.carla.actors.classify import find_hero_vehicle
from src.carla.camera.frame_buffer import CameraFrameBuffer
from src.carla.camera.sensor import configure_front_camera_blueprint, front_camera_transform
from src.carla.geometry.boxes import actor_to_gt_box
from src.carla.lidar.frame_buffer import LidarFrameBuffer
from src.carla.lidar.processing import preprocess_lidar_points
from src.carla.lidar.sensor import configure_lidar_blueprint
from src.common.runtime_config import build_live_producer_config
from src.streaming.messages import build_camera_frame_message, build_lidar_frame_message, build_state_frame_message
from src.streaming.zmq_utils import create_latest_publisher


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA live sensor producer")
    parser.add_argument("--every-nth", type=int, default=1, help="Publish every N-th CARLA frame")
    args = parser.parse_args()
    return build_live_producer_config(every_nth=args.every_nth)


def transform_to_dict(transform: carla.Transform) -> dict:
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
    config = parse_args()

    ctx = zmq.Context()
    lidar_socket = create_latest_publisher(ctx, config.lidar_bind)
    camera_socket = create_latest_publisher(ctx, config.camera_front_bind)
    state_socket = create_latest_publisher(ctx, config.state_bind)

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
    print(f"[info] ZMQ lidar: {config.lidar_bind}")
    print(f"[info] ZMQ camera_front: {config.camera_front_bind}")
    print(f"[info] ZMQ state: {config.state_bind}")
    print(f"[info] every_nth: {config.every_nth}")
    print(
        f"[info] lidar: max_range={config.lidar.max_range}, channels={config.lidar.channels}, "
        f"points_per_second={config.lidar.points_per_second}, fov=({config.lidar.lower_fov}, {config.lidar.upper_fov})"
    )
    print(
        f"[info] front camera: resolution={config.camera_front.width}x{config.camera_front.height}, "
        f"fov={config.camera_front.fov}, xyz=({config.camera_front.x}, {config.camera_front.y}, {config.camera_front.z})"
    )
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
                next_frame = last_published_frame + config.every_nth
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
                points=raw_points,
                hero=hero,
                lidar=lidar,
            )
            ego_box = actor_to_gt_box(hero, lidar_transform_snapshot).astype(np.float32)

            lidar_metadata = {
                "transform": transform_to_dict(lidar_transform_snapshot),
                "ground_z": -float(lidar_transform_relative.location.z),
            }
            camera_front_metadata = {
                "fov": float(config.camera_front.fov),
                "transform": transform_to_dict(camera_transform_snapshot),
            }

            lidar_message = build_lidar_frame_message(
                frame=int(frame_snapshot),
                timestamp=float(timestamp_snapshot),
                points=points_snapshot.astype(np.float32),
            )
            camera_message = build_camera_frame_message(
                frame=int(frame_snapshot),
                timestamp=float(timestamp_snapshot),
                camera_front_image=camera_image_bgr_snapshot,
            )
            state_message = build_state_frame_message(
                frame=int(frame_snapshot),
                timestamp=float(timestamp_snapshot),
                ego_box=ego_box,
                lidar_metadata=lidar_metadata,
                camera_front_metadata=camera_front_metadata,
            )

            lidar_socket.send_pyobj(lidar_message)
            camera_socket.send_pyobj(camera_message)
            state_socket.send_pyobj(state_message)

            last_published_frame = frame_snapshot
            published_count += 1
            print(f"[producer] frame={frame_snapshot} | published={published_count}")

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
        lidar_socket.close(0)
        camera_socket.close(0)
        state_socket.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
