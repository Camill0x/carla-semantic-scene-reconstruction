#!/usr/bin/env python3

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rerun as rr
from src.benchmark.predictions import load_lanes_prediction, load_objects_prediction
from src.carla.dataset.reader import iter_frame_dirs, load_dataset_frame
from src.common.cli_logging import configure_logging
from src.common.runtime_config import build_dataset_viewer_config, build_streaming_visualizer_config
from src.openpcdet.prediction import Objects3DPrediction
from src.rerun.dataset_viewer import initialize_dataset_viewer, log_dataset_frame
from src.rerun.lanes import log_prediction_lanes_2d, log_prediction_lanes_3d
from src.rerun.scene3d import log_prediction_objects_3d
from src.rerun.text import log_legend


@dataclass(frozen=True)
class ViewPredictionsArgs:
    run_dir: Path
    objects: Optional[Path]
    lanes: Optional[Path]
    fps: float
    show_grid: bool


def parse_args() -> ViewPredictionsArgs:
    parser = argparse.ArgumentParser(description="View offline benchmark predictions in Rerun")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to datasets/raw/run_XXXX")
    parser.add_argument("--objects", type=Path, default=None, help="Path to directory with OpenPCDet predictions")
    parser.add_argument("--lanes", type=Path, default=None, help="Path to directory with LaneDet predictions")
    parser.add_argument("--fps", type=float, default=20.0, help="Playback speed in frames per second")
    parser.add_argument("--show-grid", action="store_true", help="Show the ground grid in the 3D view")
    parsed = parser.parse_args()
    return ViewPredictionsArgs(
        run_dir=parsed.run_dir,
        objects=parsed.objects,
        lanes=parsed.lanes,
        fps=float(parsed.fps),
        show_grid=bool(parsed.show_grid),
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging("tools.benchmark.view_predictions")
    if args.fps <= 0.0:
        raise ValueError("--fps must be > 0")
    if not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)
    if not args.run_dir.is_dir():
        raise NotADirectoryError(args.run_dir)

    frame_dirs = iter_frame_dirs(args.run_dir)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* directories found in {args.run_dir}")

    dataset_config = build_dataset_viewer_config(show_grid=args.show_grid)
    live_config = build_streaming_visualizer_config(show_grid=args.show_grid)
    initialize_dataset_viewer(dataset_config)
    log_legend()

    logger.info("=== Offline prediction viewer ===")
    logger.info("run_dir: %s", args.run_dir)
    logger.info("frames: %d", len(frame_dirs))
    logger.info("fps: %.2f", args.fps)
    if args.objects:
        logger.info("objects: %s", args.objects)
    if args.lanes:
        logger.info("lanes: %s", args.lanes)

    frame_delay_s = 1.0 / args.fps
    for frame_dir in frame_dirs:
        frame = load_dataset_frame(frame_dir)
        frame_index = int(frame.meta.get("frame_index", 0))
        log_dataset_frame(frame, dataset_config)
        rr.set_time("run_time", duration=frame_index / args.fps)

        objects_path = args.objects / f"{frame_dir.name}.npz" if args.objects else None
        if objects_path and objects_path.exists():
            objects_3d = load_objects_prediction(objects_path)
            log_prediction_objects_3d(objects_3d, line_radius=live_config.pred_line_radius)
        else:
            log_prediction_objects_3d(Objects3DPrediction.empty(), line_radius=live_config.pred_line_radius)

        lanes_path = args.lanes / f"{frame_dir.name}.json" if args.lanes else None
        if lanes_path and lanes_path.exists():
            lanes_prediction = load_lanes_prediction(lanes_path)
            log_prediction_lanes_2d(lanes_prediction["lanes_2d"], line_thickness=dataset_config.lane_line_thickness)
            log_prediction_lanes_3d(lanes_prediction["lanes_3d"], line_radius=live_config.pred_line_radius)
        else:
            log_prediction_lanes_2d({"strips": []}, line_thickness=dataset_config.lane_line_thickness)
            log_prediction_lanes_3d({"strips": []}, line_radius=live_config.pred_line_radius)

        time.sleep(frame_delay_s)

    logger.info("predictions loaded into Rerun")


if __name__ == "__main__":
    main()
