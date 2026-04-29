#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
import zmq

from src.common.constants import NUSCENES_LIKE_CLASSES
from src.common.runtime_config import build_live_openpcdet_inference_config
from src.openpcdet.model import load_inference_model
from src.openpcdet.predict import filter_predictions, run_inference
from src.streaming.messages import (
    build_objects_3d_frame_message,
    parse_lidar_frame_message,
)
from src.streaming.zmq_utils import create_latest_publisher, create_latest_subscriber


def parse_args():
    parser = argparse.ArgumentParser(description="OpenPCDet live inference node")
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.2, help="Minimum score for exported predictions")
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
    logger.info("score_thresh: %.3f", config.score_thresh)
    logger.info("point_stride: %d", config.point_stride)

    context = zmq.Context()
    sub_socket = create_latest_subscriber(context, config.lidar_in)
    pub_socket = create_latest_publisher(context, config.zmq_out)

    poller = zmq.Poller()
    poller.register(sub_socket, zmq.POLLIN)

    last_frame = None
    last_log_t = time.time()
    infer_times = []
    total_times = []
    processed = 0

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

            points4 = parsed["points"]
            if config.point_stride > 1:
                points4 = points4[:: config.point_stride]

            t0 = time.time()
            try:
                pred_dict = run_inference(dataset, model, points4, frame_id)
                t1 = time.time()
                pred_boxes, pred_scores, _, pred_names = filter_predictions(
                    pred_dict=pred_dict,
                    class_names=cfg.CLASS_NAMES,
                    allowed_classes=NUSCENES_LIKE_CLASSES,
                    score_thresh=config.score_thresh,
                )
            except Exception as exc:
                logger.exception("Inference failed frame=%s: %s", frame_id, exc)
                continue

            t2 = time.time()
            out_message = build_objects_3d_frame_message(
                lidar_message=message,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_names=pred_names,
            )
            pub_socket.send_pyobj(out_message)

            last_frame = frame_id
            processed += 1
            infer_times.append(t1 - t0)
            total_times.append(time.time() - t0)

            now = time.time()
            if now - last_log_t >= 1.0 and infer_times and total_times:
                infer_ms = 1000.0 * float(np.mean(infer_times))
                total_ms = 1000.0 * float(np.mean(total_times))
                fps = processed / (now - last_log_t)
                logger.info(
                    "frame=%s | preds=%s | infer=%.1f ms | total=%.1f ms | fps=%.2f",
                    frame_id,
                    len(pred_names),
                    infer_ms,
                    total_ms,
                    fps,
                )
                infer_times = []
                total_times = []
                processed = 0
                last_log_t = now
    finally:
        sub_socket.close(0)
        pub_socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
