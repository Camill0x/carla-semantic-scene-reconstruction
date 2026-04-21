#!/usr/bin/env python3

import argparse

import numpy as np
import rerun as rr
import zmq

from src.streaming.messages import parse_lidar_message
from src.streaming.rerun_viewer import (
    initialize_viewer,
    log_gt_boxes,
    log_legend,
    log_points,
    log_prediction_boxes,
    log_status,
)
from src.streaming.zmq_utils import create_latest_subscriber


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun viewer for the live CARLA pipeline")
    parser.add_argument("--zmq-connect", default="tcp://127.0.0.1:5556")
    parser.add_argument("--show-grid", action="store_true")
    parser.add_argument("--hide-points", action="store_true")
    parser.add_argument("--hide-gt", action="store_true")
    parser.add_argument("--point-radius", type=float, default=0.03)
    parser.add_argument("--pred-line-radius", type=float, default=0.04)
    parser.add_argument("--gt-line-radius", type=float, default=0.03)
    parser.add_argument("--app-id", default="carla_live_visualizer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    context = zmq.Context()
    socket = create_latest_subscriber(context, args.zmq_connect)

    initialize_viewer(args.app_id, show_grid=args.show_grid)
    log_legend()

    try:
        while True:
            message = socket.recv_pyobj()

            while True:
                try:
                    message = socket.recv_pyobj(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

            parsed = parse_lidar_message(message)
            frame = parsed["frame"]
            points = parsed["points"]
            gt_boxes = parsed["gt_boxes"]
            gt_names = parsed["gt_names"]
            pred_boxes = np.asarray(message.get("pred_boxes", np.zeros((0, 7))), dtype=np.float32)
            pred_scores = np.asarray(message.get("pred_scores", np.zeros((0,))), dtype=np.float32)
            pred_names = [str(name) for name in message.get("pred_names", [])]

            rr.set_time("frame", sequence=frame)

            log_points(points, point_radius=args.point_radius, visible=not args.hide_points)
            log_gt_boxes(
                gt_boxes,
                gt_names,
                line_radius=args.gt_line_radius,
                visible=not args.hide_gt,
            )
            log_prediction_boxes(
                pred_boxes,
                pred_scores,
                pred_names,
                line_radius=args.pred_line_radius,
            )
            log_status(
                frame=frame,
                num_points=int(points.shape[0]),
                num_gt=int(gt_boxes.shape[0]),
                num_pred=int(pred_boxes.shape[0]),
            )
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
