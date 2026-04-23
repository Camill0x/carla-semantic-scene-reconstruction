#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import rerun as rr
from src.carla.dataset.reader import iter_frame_dirs, load_dataset_frame
from src.common.runtime_config import build_dataset_viewer_config
from src.rerun.dataset_viewer import initialize_dataset_viewer, log_dataset_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Show a saved dataset run in Rerun")
    parser.add_argument("run_dir", type=Path, help="Path to run_XXXX directory")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback speed in frames per second")
    parser.add_argument("--show-grid", action="store_true", help="Show the ground grid in the 3D view")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    print(f"[info] run dir: {args.run_dir}")
    print(f"[info] frames: {len(frame_dirs)}")
    print(f"[info] fps: {args.fps}")
    print(
        f"[info] viewer: point_radius={config.point_radius}, gt_line_radius={config.gt_line_radius}, "
        f"lane_line_thickness={config.lane_line_thickness}"
    )

    initialize_dataset_viewer(config)

    frame_delay_s = 1.0 / args.fps

    for frame_dir in frame_dirs:
        frame = load_dataset_frame(frame_dir)
        frame_index = int(frame.meta.get("frame_index", 0))
        rr.set_time("run_time", duration=frame_index / args.fps)
        log_dataset_frame(frame, config)
        time.sleep(frame_delay_s)


if __name__ == "__main__":
    main()
