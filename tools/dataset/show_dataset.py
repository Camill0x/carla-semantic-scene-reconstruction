#!/usr/bin/env python3

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import rerun as rr
from src.carla.dataset.reader import iter_frame_dirs, load_dataset_frame
from src.common.cli_logging import configure_logging
from src.common.runtime_config import build_dataset_viewer_config
from src.rerun.dataset_viewer import initialize_dataset_viewer, log_dataset_frame


@dataclass(frozen=True)
class ShowDatasetArgs:
    run_dir: Path
    fps: float
    show_grid: bool


def parse_args() -> ShowDatasetArgs:
    """Parse command-line arguments for the dataset viewer command."""
    parser = argparse.ArgumentParser(description="Show a saved dataset run in Rerun")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to datasets/raw/run_XXXX")
    parser.add_argument("--fps", type=float, default=20.0, help="Playback speed in frames per second")
    parser.add_argument("--show-grid", action="store_true", help="Show the ground grid in the 3D view")
    parsed = parser.parse_args()
    return ShowDatasetArgs(
        run_dir=parsed.run_dir,
        fps=float(parsed.fps),
        show_grid=bool(parsed.show_grid),
    )


def main() -> None:
    """Run the dataset viewer command."""
    args = parse_args()
    logger = configure_logging("tools.dataset.show_dataset")
    config = build_dataset_viewer_config(show_grid=args.show_grid)

    if args.fps <= 0.0:
        raise ValueError("--fps must be > 0")
    if not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)
    if not args.run_dir.is_dir():
        raise NotADirectoryError(args.run_dir)

    frame_dirs = iter_frame_dirs(args.run_dir)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* directories found in {args.run_dir}")

    logger.info("Run dir: %s", args.run_dir)
    logger.info("Frames: %d", len(frame_dirs))
    logger.info("FPS: %.2f", args.fps)
    logger.info(
        "Viewer: point_radius=%s, gt_line_radius=%s, lane_line_thickness=%s",
        config.point_radius,
        config.gt_line_radius,
        config.lane_line_thickness,
    )

    initialize_dataset_viewer(config)

    frame_delay_s = 1.0 / args.fps

    for frame_dir in frame_dirs:
        frame = load_dataset_frame(frame_dir)
        frame_index = int(frame.meta.get("frame_index", 0))
        rr.set_time("run_time", duration=frame_index / args.fps)
        log_dataset_frame(frame, config)
        time.sleep(frame_delay_s)

    logger.info("Dataset loaded into Rerun")


if __name__ == "__main__":
    main()
