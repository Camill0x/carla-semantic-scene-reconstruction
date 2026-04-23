#!/usr/bin/env python3

import argparse
from pathlib import Path

from src.carla.dataset.reader import load_frame
from src.carla.vis.open3d_viewer import show_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Show a saved dataset frame")
    parser.add_argument("frame_dir", type=Path, help="Path to frame directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta, points, ego_box, objects, lanes = load_frame(args.frame_dir)
    lidar_transform = {
        "location": meta["lidar"]["location"],
        "rotation": meta["lidar"]["rotation"],
    }

    print("[info] frame_index:", meta["frame_index"])
    print("[info] sim_frame:", meta["sim_frame"])
    print("[info] points shape:", points.shape)
    print("[info] num_objects:", len(objects))
    print("[info] num_lanes:", len(lanes))

    show_frame(lidar_transform, points, ego_box, objects)


if __name__ == "__main__":
    main()
