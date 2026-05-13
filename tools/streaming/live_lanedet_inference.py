#!/usr/bin/env python3

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.common.cli_logging import print_verbose
from src.common.runtime_config import build_streaming_lanedet_inference_config
from src.lanedet.model import LaneDetector
from src.lanedet.projection import lanes_2d_to_lanes_3d
from src.shared_memory.buffers import SharedArrayReader, SharedMessageBuffer
from src.shared_memory.names import SharedMemoryNames
from src.streaming.messages import build_lanes_3d_frame_message, parse_frame_snapshot_message


@dataclass(frozen=True)
class LiveLaneDetArgs:
    config: Path
    ckpt: Path
    score_thresh: float
    verbose: bool


def parse_args() -> LiveLaneDetArgs:
    parser = argparse.ArgumentParser(description="LaneDet live inference node")
    parser.add_argument("--config", type=Path, required=True, help="LaneDet config file")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.2, help="Score threshold for predictions")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame logs")
    parsed = parser.parse_args()
    return LiveLaneDetArgs(
        config=parsed.config,
        ckpt=parsed.ckpt,
        score_thresh=float(parsed.score_thresh),
        verbose=bool(parsed.verbose),
    )


def main() -> None:
    args = parse_args()
    config = build_streaming_lanedet_inference_config(
        cfg_file=args.config.expanduser().resolve(),
        ckpt=args.ckpt.expanduser().resolve(),
        score_thresh=args.score_thresh,
    )
    names = SharedMemoryNames(prefix=config.common.prefix)
    detector = LaneDetector(config.cfg_file, config.ckpt, score_thresh=config.score_thresh)

    frame_buffer = SharedMessageBuffer(
        name=names.frame_buffer,
        size_bytes=config.common.frame_buffer_size_bytes,
        create=False,
    )
    lanes_buffer = SharedMessageBuffer(
        name=names.lanes_buffer,
        size_bytes=config.common.lanes_buffer_size_bytes,
        create=False,
    )
    reader = SharedArrayReader()
    last_version = None
    last_frame = None
    sleep_s = max(0.001, config.common.poll_interval_ms / 1000.0)

    print("=== LaneDet streaming inference ===")
    print(f"[info] frame buffer: {names.frame_buffer}")
    print(f"[info] lanes buffer: {names.lanes_buffer}")

    try:
        while True:
            version, payload = frame_buffer.read(last_version=last_version)
            if payload is None:
                time.sleep(sleep_s)
                continue

            last_version = version
            try:
                if not isinstance(payload, Mapping):
                    raise ValueError("Frame payload is not a mapping")
                frame_message = parse_frame_snapshot_message(payload)
            except Exception as exc:
                print(f"[warn] invalid frame snapshot: {exc}")
                continue

            frame_id = int(frame_message["frame"])
            if frame_id == last_frame:
                continue

            try:
                image_bgr = reader.read(frame_message["camera_front"]["shared_array"]).astype(np.uint8, copy=False)
                camera_frame = {
                    "frame": frame_message["frame"],
                    "timestamp": frame_message["timestamp"],
                    "camera_front": {
                        **frame_message["camera_front"],
                        "image": image_bgr,
                    },
                }
                lanes_2d = detector.infer_lanes_2d(image_bgr)
                lanes_3d = lanes_2d_to_lanes_3d(
                    lanes_2d,
                    camera_frame=camera_frame,
                    state_frame=frame_message["state"],
                    score_thresh=config.score_thresh,
                )
            except Exception as exc:
                print(f"[warn] LaneDet inference failed frame={frame_id}: {exc}")
                continue

            lanes_buffer.write(
                build_lanes_3d_frame_message(
                    camera_message=frame_message,
                    lanes_3d=lanes_3d,
                )
            )
            print_verbose(args.verbose, "LaneDet", f"Detected {len(lanes_3d)} lanes for frame {frame_id}")
            last_frame = frame_id
    finally:
        reader.close()
        frame_buffer.close()
        lanes_buffer.close()


if __name__ == "__main__":
    main()
