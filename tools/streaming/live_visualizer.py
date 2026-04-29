#!/usr/bin/env python3

import argparse

import zmq

from src.common.runtime_config import build_live_visualizer_config
from src.rerun.live_viewer import initialize_live_viewer, LiveRenderState
from src.rerun.text import log_legend
from src.streaming.messages import (
    parse_gt_frame_message,
    parse_lanes_3d_frame_message,
    parse_lidar_frame_message,
    parse_objects_3d_frame_message,
    parse_state_frame_message,
)
from src.streaming.zmq_utils import create_latest_subscriber


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun viewer for the live CARLA pipeline")
    parser.add_argument("--show-grid", action="store_true", help="Show the ground grid")
    args = parser.parse_args()
    config = build_live_visualizer_config(
        show_grid=args.show_grid,
    )
    return config


def drain_latest(socket: zmq.Socket):
    message = socket.recv_pyobj()
    while True:
        try:
            message = socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return message


def main() -> None:
    config = parse_args()

    context = zmq.Context()
    objects_socket = create_latest_subscriber(context, config.objects_3d_connect)
    lanes_socket = create_latest_subscriber(context, config.lanes_3d_connect)
    lidar_socket = create_latest_subscriber(context, config.lidar_connect)
    state_socket = create_latest_subscriber(context, config.state_connect)
    gt_socket = create_latest_subscriber(context, config.gt_connect)
    poller = zmq.Poller()
    poller.register(objects_socket, zmq.POLLIN)
    poller.register(lanes_socket, zmq.POLLIN)
    poller.register(lidar_socket, zmq.POLLIN)
    poller.register(state_socket, zmq.POLLIN)
    poller.register(gt_socket, zmq.POLLIN)

    initialize_live_viewer("carla_live_visualizer", show_grid=config.show_grid)
    log_legend()

    render_state = LiveRenderState()

    try:
        while True:
            events = dict(poller.poll(timeout=100))

            if objects_socket in events:
                try:
                    render_state.update_objects(parse_objects_3d_frame_message(drain_latest(objects_socket)))
                except Exception as exc:
                    print(f"[warn] invalid objects_3d message: {exc}")

            if lanes_socket in events:
                try:
                    render_state.update_lanes(parse_lanes_3d_frame_message(drain_latest(lanes_socket)))
                except Exception as exc:
                    print(f"[warn] invalid lanes_3d message: {exc}")

            if lidar_socket in events:
                try:
                    render_state.update_lidar(parse_lidar_frame_message(drain_latest(lidar_socket)))
                except Exception as exc:
                    print(f"[warn] invalid lidar_frame message: {exc}")

            if state_socket in events:
                try:
                    render_state.update_state(parse_state_frame_message(drain_latest(state_socket)))
                except Exception as exc:
                    print(f"[warn] invalid state_frame message: {exc}")

            if gt_socket in events:
                try:
                    render_state.update_gt(parse_gt_frame_message(drain_latest(gt_socket)))
                except Exception as exc:
                    print(f"[warn] invalid gt_frame message: {exc}")

            render_state.render_next(config)
    finally:
        objects_socket.close(0)
        lanes_socket.close(0)
        lidar_socket.close(0)
        state_socket.close(0)
        gt_socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
