#!/usr/bin/env python3

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import zmq

from src.common.cli_logging import configure_logging
from src.common.runtime_config import build_streaming_visualizer_config
from src.rerun.live_viewer import initialize_live_viewer, render_live_scene
from src.rerun.text import log_legend
from src.shared_memory.buffers import measure_json_payload_bytes
from src.streaming.messages import parse_scene_frame_message
from src.streaming.zmq_utils import create_latest_subscriber, drain_latest


@dataclass(frozen=True)
class LiveVisualizerArgs:
    show_grid: bool
    verbose: bool


def parse_args() -> LiveVisualizerArgs:
    """Parse command-line arguments for the live streaming visualizer."""
    parser = argparse.ArgumentParser(description="Rerun viewer for the live streaming pipeline")
    parser.add_argument("--show-grid", action="store_true", help="Show the ground grid")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame logs")
    parsed = parser.parse_args()
    return LiveVisualizerArgs(
        show_grid=bool(parsed.show_grid),
        verbose=bool(parsed.verbose),
    )


def main() -> None:
    """Run the live streaming visualizer."""
    args = parse_args()
    logger = configure_logging("tools.streaming.live_visualizer", verbose=args.verbose)
    config = build_streaming_visualizer_config(show_grid=args.show_grid)

    context = zmq.Context()
    scene_socket = create_latest_subscriber(context, config.scene_connect)
    poller = zmq.Poller()
    poller.register(scene_socket, zmq.POLLIN)

    initialize_live_viewer("carla_streaming_visualizer", show_grid=config.show_grid)
    log_legend()
    latest_scene = None
    last_render_key: Optional[Tuple[int, object, object, object]] = None

    logger.info("=== Streaming visualizer ===")
    logger.info("ZMQ scene IN: %s", config.scene_connect)

    try:
        while True:
            events = dict(poller.poll(timeout=100))
            if scene_socket in events:
                try:
                    receive_monotonic_ns = time.monotonic_ns()
                    raw_scene = drain_latest(scene_socket)
                    transfer_bytes = measure_json_payload_bytes(raw_scene)
                    scene = parse_scene_frame_message(raw_scene)
                    published_at_ns = int(scene.get("published_at_monotonic_ns", 0))
                    scene["latency_ms"] = (receive_monotonic_ns - published_at_ns) / 1_000_000.0
                    scene["transfer_bytes"] = transfer_bytes
                    logger.debug("received_frame=%s", scene["frame"])
                    latest_scene = scene
                except Exception as exc:
                    logger.warning("Invalid scene_frame: %s", exc)

            if latest_scene is not None:
                source_frames = latest_scene.get("source_frames", {})
                render_key = (
                    int(latest_scene["frame"]),
                    source_frames.get("objects_3d"),
                    source_frames.get("lanes_3d"),
                    source_frames.get("state"),
                )
                if render_key != last_render_key:
                    render_live_scene(
                        latest_scene,
                        ego_line_radius=config.ego_line_radius,
                        pred_line_radius=config.pred_line_radius,
                    )
                    last_render_key = render_key
    finally:
        scene_socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
