#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import zmq

from src.common.runtime_config import build_live_lanedet_inference_config
from src.lanedet.model import LaneDetector
from src.lanedet.projection import lanes_2d_to_lanes_3d
from src.streaming.messages import (
    build_lanes_3d_frame_message,
    parse_camera_frame_message,
    parse_state_frame_message,
)
from src.streaming.zmq_utils import create_latest_publisher, create_latest_subscriber, drain_latest


def parse_args():
    parser = argparse.ArgumentParser(description="LaneDet live inference node")
    parser.add_argument("--config", type=Path, required=True, help="LaneDet config file")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.2, help="Score threshold for predictions")
    args = parser.parse_args()
    config = build_live_lanedet_inference_config(
        cfg_file=args.config.expanduser().resolve(),
        ckpt=args.ckpt.expanduser().resolve(),
        score_thresh=args.score_thresh,
    )
    return config


def main() -> None:
    config = parse_args()
    detector = LaneDetector(
        config.cfg_file,
        config.ckpt,
        score_thresh=config.score_thresh,
    )

    print("=== LaneDet live inference ===")
    print(f"[info] ZMQ camera_front IN: {config.camera_front_in}")
    print(f"[info] ZMQ state IN: {config.state_in}")
    print(f"[info] ZMQ OUT: {config.zmq_out}")
    print(f"[info] config: {config.cfg_file}")
    print(f"[info] ckpt: {config.ckpt}")
    print(f"[info] score_thresh: {config.score_thresh:.2f}")

    context = zmq.Context()
    camera_socket = create_latest_subscriber(context, config.camera_front_in)
    state_socket = create_latest_subscriber(context, config.state_in)
    pub_socket = create_latest_publisher(context, config.zmq_out)

    poller = zmq.Poller()
    poller.register(camera_socket, zmq.POLLIN)
    poller.register(state_socket, zmq.POLLIN)

    latest_camera = None
    latest_state = None
    last_frame = None

    try:
        while True:
            events = dict(poller.poll(timeout=50))
            if not events:
                continue

            if camera_socket in events:
                try:
                    latest_camera = parse_camera_frame_message(drain_latest(camera_socket))
                except Exception as exc:
                    print(f"[warn] invalid camera_frame: {exc}")

            if state_socket in events:
                try:
                    latest_state = parse_state_frame_message(drain_latest(state_socket))
                except Exception as exc:
                    print(f"[warn] invalid state_frame: {exc}")

            if latest_camera is None or latest_state is None:
                continue
            if latest_camera["frame"] != latest_state["frame"]:
                continue

            frame_id = int(latest_camera["frame"])
            if frame_id == last_frame:
                continue

            camera = latest_camera["camera_front"]
            image_bgr = np.asarray(camera["image"], dtype=np.uint8)
            try:
                lanes_2d = detector.infer_lanes_2d(image_bgr)
                lanes_3d = lanes_2d_to_lanes_3d(
                    lanes_2d,
                    camera_frame=latest_camera,
                    state_frame=latest_state,
                    score_thresh=config.score_thresh,
                )
            except Exception as exc:
                print(f"[warn] LaneDet inference failed frame={frame_id}: {exc}")
                continue

            out_message = build_lanes_3d_frame_message(
                camera_message=latest_camera,
                lanes_3d=lanes_3d,
            )
            pub_socket.send_pyobj(out_message)

            last_frame = frame_id
    finally:
        camera_socket.close(0)
        state_socket.close(0)
        pub_socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
