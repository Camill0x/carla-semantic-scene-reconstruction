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
    points, meta, ego_box = load_frame(args.frame_dir)

    print("[info] frame:", meta.get("frame"))
    print("[info] points shape:", points.shape)
    print("[info] num_objects:", len(meta.get("objects", [])))
    if "class_counts" in meta:
        print("[info] class_counts:", meta["class_counts"])

    show_frame(points, meta, ego_box)


if __name__ == "__main__":
    main()
