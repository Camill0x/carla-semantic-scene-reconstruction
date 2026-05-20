#!/usr/bin/env python3

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.common.cli_logging import configure_logging
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
    """Parse command-line arguments for the live LaneDet inference node."""
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
    """Run the live LaneDet inference node."""
    args = parse_args()
    logger = configure_logging("tools.streaming.live_lanedet_inference", verbose=args.verbose)
    config = build_streaming_lanedet_inference_config(
        cfg_file=args.config.expanduser().resolve(),
        ckpt=args.ckpt.expanduser().resolve(),
        score_thresh=args.score_thresh,
    )
    names = SharedMemoryNames(prefix=config.common.prefix)
    detector = LaneDetector(config.cfg_file, config.ckpt, score_thresh=config.score_thresh, logger=logger)

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

    logger.info("=== LaneDet streaming inference ===")
    logger.info("Frame buffer: %s", names.frame_buffer)
    logger.info("Lanes buffer: %s", names.lanes_buffer)

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
                logger.warning("Invalid frame snapshot: %s", exc)
                continue

            frame_id = int(frame_message["frame"])
            if frame_id == last_frame:
                continue

            try:
                image_bgr = reader.read(frame_message["camera_front"]["shared_array"])
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
                logger.warning("LaneDet inference failed frame=%s: %s", frame_id, exc)
                continue

            lanes_buffer.write(
                build_lanes_3d_frame_message(
                    camera_message=frame_message,
                    lanes_3d=lanes_3d,
                )
            )
            logger.debug("frame=%s | predicted_lanes=%d", frame_id, len(lanes_3d))
            last_frame = frame_id
    finally:
        reader.close()
        frame_buffer.close()
        lanes_buffer.close()


if __name__ == "__main__":
    main()
