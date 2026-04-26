#!/usr/bin/env python3

import argparse
import queue

import carla
from src.carla.actors.classify import find_hero_vehicle
from src.carla.camera.frame_buffer import CameraFrameBuffer
from src.carla.camera.sensor import (
    configure_front_camera_blueprint,
    front_camera_transform,
)
from src.carla.dataset.paths import ensure_dataset_run_dir
from src.carla.dataset.writer import save_multimodal_frame
from src.carla.gt.collector import collect_gt, count_by_class
from src.carla.gt.filtering import filter_gt
from src.carla.lanes.collector import collect_lane_annotations
from src.carla.lidar.frame_buffer import LidarFrameBuffer
from src.carla.lidar.processing import preprocess_lidar_points
from src.carla.lidar.sensor import configure_lidar_blueprint
from src.common.runtime_config import build_collector_config


def parse_args():
    parser = argparse.ArgumentParser(description="Collect raw multimodal CARLA dataset frames")
    parser.add_argument("-n", "--num-frames", type=int, default=100, help="How many frames to save")
    parser.add_argument("--every-nth", type=int, default=10, help="Save every N-th synchronized world frame")
    args = parser.parse_args()
    return build_collector_config(
        num_frames=args.num_frames,
        every_nth=args.every_nth,
    )


def main() -> None:
    config = parse_args()

    client = carla.Client(config.carla.host, config.carla.port)
    client.set_timeout(5.0)
    world = client.get_world()

    settings = world.get_settings()
    if not settings.synchronous_mode:
        raise RuntimeError("Run manual_control.py with the --sync flag")

    hero = find_hero_vehicle(world)
    if hero is None:
        raise RuntimeError("Hero vehicle not found")

    run_dir = ensure_dataset_run_dir(config.dataset_root_dir)

    print(f"[info] hero id={hero.id}, type={hero.type_id}")
    print(f"[info] run dir: {run_dir}")
    print(f"[info] num_frames: {config.num_frames}")
    print(f"[info] every_nth: {config.every_nth}")
    print("[info] sensors: lidar + front camera")
    print(
        f"[info] lidar: max_range={config.lidar.max_range}m, channels={config.lidar.channels}, "
        f"points_per_second={config.lidar.points_per_second}, fov=({config.lidar.lower_fov}, {config.lidar.upper_fov})"
    )
    print(
        f"[info] front camera: resolution={config.camera_front.width}x{config.camera_front.height}, "
        f"fov={config.camera_front.fov}, xyz=({config.camera_front.x}, {config.camera_front.y}, {config.camera_front.z})"
    )
    print(
        f"[info] lane annotations: distance={config.lane_annotations.distance_m}m, "
        f"step={config.lane_annotations.step_m}m, max_side_lanes={config.lane_annotations.max_side_lanes}"
    )
    print(f"[info] gt annotations: min_lidar_points_in_box={config.gt_annotations.min_lidar_points_in_box}")

    lidar_bp = configure_lidar_blueprint(
        world,
        config=config.lidar,
        fixed_delta_seconds=settings.fixed_delta_seconds,
    )
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

        saved_count = 0
        last_saved_frame = None

        while saved_count < config.num_frames:
            snapshot = world.wait_for_tick()
            expected_frame = int(snapshot.frame) if hasattr(snapshot, "frame") else int(snapshot)

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
                if saved_count == 0:
                    raise RuntimeError("Timeout: no input data received")
                continue

            raw_points, sim_frame, lidar_timestamp, lidar_transform_snapshot = lidar_data
            camera_image_bgr_snapshot, _camera_frame, camera_timestamp, camera_transform_snapshot = camera_data
            timestamp_snapshot = max(lidar_timestamp, camera_timestamp)

            if last_saved_frame is not None and sim_frame == last_saved_frame:
                continue

            if sim_frame % config.every_nth != 0:
                continue

            points_snapshot = preprocess_lidar_points(
                points=raw_points,
                hero=hero,
                lidar=lidar,
            )

            objects, gt_boxes, gt_names = collect_gt(
                world=world,
                hero=hero,
                lidar_transform=lidar_transform_snapshot,
                max_range=config.lidar.max_range,
            )

            objects, gt_boxes, gt_names = filter_gt(
                points=points_snapshot,
                objects=objects,
                gt_boxes=gt_boxes,
                gt_names=gt_names,
                min_points_in_box=config.gt_annotations.min_lidar_points_in_box,
            )

            class_counts = count_by_class(objects)
            lane_annotations = collect_lane_annotations(
                world=world,
                hero=hero,
                lidar_transform=lidar_transform_snapshot,
                camera_transform=camera_transform_snapshot,
                image_width=config.camera_front.width,
                image_height=config.camera_front.height,
                camera_fov=config.camera_front.fov,
                config=config.lane_annotations,
            )

            save_multimodal_frame(
                output_root=run_dir,
                frame_index=saved_count,
                sim_frame=sim_frame,
                timestamp=timestamp_snapshot,
                world=world,
                hero=hero,
                lidar_transform=lidar_transform_snapshot,
                camera_transform=camera_transform_snapshot,
                points=points_snapshot,
                image_bgr=camera_image_bgr_snapshot,
                lanes=lane_annotations,
                objects=objects,
                class_counts=class_counts,
                gt_boxes=gt_boxes,
                gt_names=gt_names,
                config=config,
            )

            print(
                f"[saved {saved_count + 1}/{config.num_frames}] "
                f"sim_frame={sim_frame} | points={points_snapshot.shape[0]} | "
                f"lanes={len(lane_annotations)} | objects={len(objects)}"
            )

            last_saved_frame = sim_frame
            saved_count += 1

        print(f"[done] raw dataset collection finished: {run_dir}")

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


if __name__ == "__main__":
    main()
