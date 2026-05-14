#!/usr/bin/env python3

import argparse
import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np

import carla
from src.carla.actors.classify import wait_for_hero_vehicle
from src.carla.camera.frame_buffer import CameraFrameBuffer
from src.carla.camera.sensor import configure_front_camera_blueprint, front_camera_transform
from src.carla.dataset.paths import ensure_dataset_run_dir
from src.carla.dataset.writer import save_multimodal_frame
from src.carla.gt.collector import collect_gt, count_by_class
from src.carla.gt.filtering import filter_gt
from src.carla.lanes.collector import collect_lane_annotations
from src.carla.lidar.frame_buffer import LidarFrameBuffer
from src.carla.lidar.processing import preprocess_lidar_points
from src.carla.lidar.sensor import configure_lidar_blueprint
from src.common.cli_logging import configure_logging
from src.common.runtime_config import build_collector_config


@dataclass(frozen=True)
class CollectDatasetArgs:
    num_frames: int
    every_nth: int


def parse_args() -> CollectDatasetArgs:
    """Parse command-line arguments for the raw dataset collection command."""
    parser = argparse.ArgumentParser(description="Collect raw multimodal CARLA dataset frames")
    parser.add_argument("-n", "--num-frames", type=int, default=100, help="How many frames to save")
    parser.add_argument("--every-nth", type=int, default=10, help="Save every N-th synchronized world frame")
    parsed = parser.parse_args()
    return CollectDatasetArgs(
        num_frames=int(parsed.num_frames),
        every_nth=int(parsed.every_nth),
    )


def should_process_frame(frame: int, last_processed_frame: Optional[int], every_nth: int) -> bool:
    """Return whether the current frame should be processed for the configured sampling interval."""
    return last_processed_frame is None or frame - last_processed_frame >= every_nth


def main() -> None:
    """Run the raw dataset collection command."""
    args = parse_args()
    logger = configure_logging("tools.dataset.collect_dataset")
    config = build_collector_config(
        num_frames=args.num_frames,
        every_nth=args.every_nth,
    )

    client = carla.Client(config.carla.host, config.carla.port)
    client.set_timeout(5.0)
    logger.info("Waiting for a synchronized CARLA world with an active hero vehicle...")
    world, hero = wait_for_hero_vehicle(client, require_sync=True)
    settings = world.get_settings()

    run_dir = ensure_dataset_run_dir(config.dataset_root_dir)

    logger.info("Hero id=%s, type=%s", hero.id, hero.type_id)
    logger.info("Map: %s", world.get_map().name)
    logger.info("Run dir: %s", run_dir)
    logger.info("Num frames: %d", config.num_frames)
    logger.info("Every nth: %d", config.every_nth)
    logger.info("Sensors: lidar + front camera")
    logger.info(
        "Lidar: max_range=%sm, channels=%s, points_per_second=%s, fov=(%s, %s)",
        config.lidar.max_range,
        config.lidar.channels,
        config.lidar.points_per_second,
        config.lidar.lower_fov,
        config.lidar.upper_fov,
    )
    logger.info(
        "Front camera: resolution=%sx%s, fov=%s, xyz=(%s, %s, %s)",
        config.camera_front.width,
        config.camera_front.height,
        config.camera_front.fov,
        config.camera_front.x,
        config.camera_front.y,
        config.camera_front.z,
    )
    logger.info(
        "Lane annotations: distance=%sm, step=%sm, max_side_lanes=%s",
        config.lane_annotations.distance_m,
        config.lane_annotations.step_m,
        config.lane_annotations.max_side_lanes,
    )
    logger.info(
        "GT annotations: min_lidar_points_in_box=%s",
        config.gt_annotations.min_lidar_points_in_box,
    )

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

            if not should_process_frame(expected_frame, last_saved_frame, config.every_nth):
                next_frame = config.every_nth if last_saved_frame is None else last_saved_frame + config.every_nth
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
                if saved_count == 0:
                    raise RuntimeError("Timeout: no input data received")
                continue

            raw_points, sim_frame, lidar_timestamp, lidar_transform_snapshot = lidar_data
            camera_image_bgr_snapshot, _camera_frame, camera_timestamp, camera_transform_snapshot = camera_data
            timestamp_snapshot = max(lidar_timestamp, camera_timestamp)

            if last_saved_frame is not None and sim_frame == last_saved_frame:
                continue

            if not should_process_frame(sim_frame, last_saved_frame, config.every_nth):
                continue

            points_snapshot = preprocess_lidar_points(
                points=np.asarray(raw_points, dtype=np.float64),
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
                points=np.asarray(points_snapshot, dtype=np.float32),
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

            logger.info(
                "saved %d/%d | sim_frame=%s | points=%d | lanes=%d | objects=%d",
                saved_count + 1,
                config.num_frames,
                sim_frame,
                points_snapshot.shape[0],
                len(lane_annotations),
                len(objects),
            )

            last_saved_frame = sim_frame
            saved_count += 1

        logger.info("Raw dataset collection finished: %s", run_dir)

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
