#!/usr/bin/env python3

import argparse

import numpy as np
import rerun as rr
import zmq

from src.common.runtime_config import build_live_visualizer_config
from src.rerun.live_viewer import initialize_live_viewer, log_legend, log_live_status
from src.rerun.scene3d import log_gt_boxes, log_points, log_prediction_boxes
from src.streaming.messages import parse_lidar_message
from src.streaming.zmq_utils import create_latest_subscriber


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun viewer for the live CARLA pipeline")
    parser.add_argument("--show-grid", action="store_true", help="Show the ground grid")
    parser.add_argument("--hide-points", action="store_true", help="Hide point cloud")
    parser.add_argument("--hide-gt", action="store_true", help="Hide ground-truth boxes")
    args = parser.parse_args()
    return build_live_visualizer_config(
        show_grid=args.show_grid,
        hide_points=args.hide_points,
        hide_gt=args.hide_gt,
    )


def main() -> None:
    config = parse_args()

    context = zmq.Context()
    socket = create_latest_subscriber(context, config.zmq_connect)

    initialize_live_viewer("carla_live_visualizer", show_grid=config.show_grid)
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

            log_points(points, point_radius=config.point_radius, visible=not config.hide_points)
            log_gt_boxes(
                gt_boxes,
                gt_names,
                line_radius=config.gt_line_radius,
                visible=not config.hide_gt,
            )
            log_prediction_boxes(
                pred_boxes,
                pred_scores,
                pred_names,
                line_radius=config.pred_line_radius,
            )
            log_live_status(
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
