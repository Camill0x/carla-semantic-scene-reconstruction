#!/usr/bin/env python3

import argparse
import time
from typing import Dict, Optional

import zmq

from src.common.cli_logging import print_verbose
from src.common.runtime_config import build_streaming_aggregator_config
from src.shared_memory.buffers import SharedMessageBuffer
from src.shared_memory.names import SharedMemoryNames
from src.streaming.messages import (
    build_scene_frame_message,
    parse_frame_snapshot_message,
    parse_lanes_3d_frame_message,
    parse_objects_3d_frame_message,
)
from src.streaming.zmq_utils import create_latest_publisher


def parse_args():
    parser = argparse.ArgumentParser(description="Scene aggregator for the live streaming pipeline")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame logs")
    return parser.parse_args()


def trim_cache(cache: Dict[int, dict], limit: int) -> None:
    while len(cache) > limit:
        del cache[min(cache)]


def try_open_buffer(name: str, size_bytes: int) -> Optional[SharedMessageBuffer]:
    try:
        return SharedMessageBuffer(name=name, size_bytes=size_bytes, create=False)
    except FileNotFoundError:
        return None


def next_ready_frame(
    *,
    frame_cache: Dict[int, dict],
    objects_frames: Dict[int, dict],
    lanes_frames: Dict[int, dict],
    require_objects: bool,
    require_lanes: bool,
    last_sent_frame: Optional[int],
) -> Optional[int]:
    for frame in sorted(frame_cache):
        if last_sent_frame is not None and frame <= last_sent_frame:
            continue
        if require_objects and frame not in objects_frames:
            continue
        if require_lanes and frame not in lanes_frames:
            continue
        return frame
    return None


def main() -> None:
    args = parse_args()
    config = build_streaming_aggregator_config()
    names = SharedMemoryNames(prefix=config.common.prefix)
    frame_buffer = None
    objects_buffer = None
    lanes_buffer = None

    context = zmq.Context()
    scene_socket = create_latest_publisher(context, config.scene_bind)

    objects_frames: Dict[int, dict] = {}
    lanes_frames: Dict[int, dict] = {}
    frame_cache: Dict[int, dict] = {}
    frame_cache_limit = 120
    frame_version = None
    objects_version = None
    lanes_version = None
    objects_active = False
    lanes_active = False
    objects_last_update_t = None
    lanes_last_update_t = None
    stale_timeout_s = 3.0
    last_sent_frame = None
    sleep_s = max(0.001, config.common.poll_interval_ms / 1000.0)

    print("=== Streaming scene aggregator ===")
    print(f"[info] frame buffer: {names.frame_buffer}")
    print(f"[info] objects buffer: {names.objects_buffer}")
    print(f"[info] lanes buffer: {names.lanes_buffer}")
    print(f"[info] ZMQ scene OUT: {config.scene_bind}")

    try:
        while True:
            updated = False
            now = time.monotonic()

            if frame_buffer is None:
                frame_buffer = try_open_buffer(
                    names.frame_buffer,
                    config.common.frame_buffer_size_bytes,
                )
                if frame_buffer is None:
                    time.sleep(sleep_s)
                    continue

            if objects_buffer is None:
                objects_buffer = try_open_buffer(
                    names.objects_buffer,
                    config.common.objects_buffer_size_bytes,
                )

            if lanes_buffer is None:
                lanes_buffer = try_open_buffer(
                    names.lanes_buffer,
                    config.common.lanes_buffer_size_bytes,
                )

            new_frame_version, frame_payload = frame_buffer.read(last_version=frame_version)
            if frame_payload is not None:
                frame_version = new_frame_version
                try:
                    frame_message = parse_frame_snapshot_message(frame_payload)
                    frame_cache[int(frame_message["frame"])] = frame_message
                    trim_cache(frame_cache, frame_cache_limit)
                    updated = True
                except Exception as exc:
                    print(f"[warn] invalid frame snapshot: {exc}")

            if objects_buffer is not None:
                new_objects_version, objects_payload = objects_buffer.read(last_version=objects_version)
                if objects_payload is not None:
                    objects_version = new_objects_version
                    try:
                        message = parse_objects_3d_frame_message(objects_payload)
                        objects_frames[int(message["frame"])] = message
                        trim_cache(objects_frames, frame_cache_limit)
                        objects_active = True
                        objects_last_update_t = now
                        updated = True
                    except Exception as exc:
                        print(f"[warn] invalid objects frame: {exc}")

            if lanes_buffer is not None:
                new_lanes_version, lanes_payload = lanes_buffer.read(last_version=lanes_version)
                if lanes_payload is not None:
                    lanes_version = new_lanes_version
                    try:
                        message = parse_lanes_3d_frame_message(lanes_payload)
                        lanes_frames[int(message["frame"])] = message
                        trim_cache(lanes_frames, frame_cache_limit)
                        lanes_active = True
                        lanes_last_update_t = now
                        updated = True
                    except Exception as exc:
                        print(f"[warn] invalid lanes frame: {exc}")

            if objects_active and objects_last_update_t is not None and now - objects_last_update_t > stale_timeout_s:
                objects_active = False
                objects_last_update_t = None
                objects_frames.clear()
                updated = True
                print("[aggregator] objects branch inactive; continuing without objects")

            if lanes_active and lanes_last_update_t is not None and now - lanes_last_update_t > stale_timeout_s:
                lanes_active = False
                lanes_last_update_t = None
                lanes_frames.clear()
                updated = True
                print("[aggregator] lanes branch inactive; continuing without lanes")

            if not frame_cache:
                time.sleep(sleep_s)
                continue

            if not updated:
                time.sleep(sleep_s)
                continue

            frame = next_ready_frame(
                frame_cache=frame_cache,
                objects_frames=objects_frames,
                lanes_frames=lanes_frames,
                require_objects=objects_active,
                require_lanes=lanes_active,
                last_sent_frame=last_sent_frame,
            )
            if frame is None:
                time.sleep(sleep_s)
                continue

            frame_message = frame_cache[frame]
            objects_message = objects_frames.get(frame)
            lanes_message = lanes_frames.get(frame)
            scene = build_scene_frame_message(
                frame_message=frame_message,
                objects_message=objects_message,
                lanes_message=lanes_message,
            )

            scene_socket.send_pyobj(scene)
            print_verbose(args.verbose, "Aggregator", f"Published frame {scene['frame']}")
            last_sent_frame = frame
    finally:
        if frame_buffer is not None:
            frame_buffer.close()
        if objects_buffer is not None:
            objects_buffer.close()
        if lanes_buffer is not None:
            lanes_buffer.close()
        scene_socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
