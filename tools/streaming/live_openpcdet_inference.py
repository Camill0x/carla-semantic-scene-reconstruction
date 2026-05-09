#!/usr/bin/env python3

import argparse
from pathlib import Path

import zmq

from src.common.constants import NUSCENES_LIKE_CLASSES
from src.common.runtime_config import build_live_openpcdet_inference_config
from src.openpcdet.model import load_inference_model, run_inference
from src.openpcdet.postprocess import filter_object_predictions
from src.streaming.messages import build_objects_3d_frame_message, parse_lidar_frame_message
from src.streaming.zmq_utils import create_latest_publisher, create_latest_subscriber


def parse_args():
    parser = argparse.ArgumentParser(description="OpenPCDet live inference node")
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.05, help="Score threshold for predictions")
    parser.add_argument("--point-stride", type=int, default=1, help="Take every N-th point before inference")
    args = parser.parse_args()
    config = build_live_openpcdet_inference_config(
        cfg_file=args.cfg_file,
        ckpt=args.ckpt,
        score_thresh=args.score_thresh,
        point_stride=args.point_stride,
    )
    return config


def main() -> None:
    config = parse_args()

    dataset, model, cfg, logger = load_inference_model(config.cfg_file, config.ckpt)

    logger.info("=== OpenPCDet live inference ===")
    logger.info("ZMQ lidar IN: %s", config.lidar_in)
    logger.info("ZMQ OUT: %s", config.zmq_out)
    logger.info("score_thresh: %.2f", config.score_thresh)
    logger.info("point_stride: %d", config.point_stride)

    context = zmq.Context()
    sub_socket = create_latest_subscriber(context, config.lidar_in)
    pub_socket = create_latest_publisher(context, config.zmq_out)

    poller = zmq.Poller()
    poller.register(sub_socket, zmq.POLLIN)

    last_frame = None

    try:
        while True:
            events = dict(poller.poll(timeout=50))
            if sub_socket not in events:
                continue

            try:
                message = sub_socket.recv_pyobj()
                while True:
                    try:
                        message = sub_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
            except Exception as exc:
                logger.exception("ZMQ receive failed: %s", exc)
                continue

            try:
                parsed = parse_lidar_frame_message(message)
            except Exception as exc:
                logger.warning("Skipping invalid input message: %s", exc)
                continue

            frame_id = parsed["frame"]
            if frame_id == last_frame:
                continue

            try:
                points4 = parsed["points"]
                if config.point_stride > 1:
                    points4 = points4[:: config.point_stride]

                raw_objects_3d = run_inference(dataset, model, points4, frame_id)
                objects_3d = filter_object_predictions(
                    objects_3d=raw_objects_3d,
                    class_names=cfg.CLASS_NAMES,
                    allowed_classes=NUSCENES_LIKE_CLASSES,
                    score_thresh=config.score_thresh,
                )
            except Exception as exc:
                logger.exception("Inference failed frame=%s: %s", frame_id, exc)
                continue

            out_message = build_objects_3d_frame_message(
                lidar_message=message,
                objects_3d=objects_3d,
            )
            pub_socket.send_pyobj(out_message)

            last_frame = frame_id
    finally:
        sub_socket.close(0)
        pub_socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
