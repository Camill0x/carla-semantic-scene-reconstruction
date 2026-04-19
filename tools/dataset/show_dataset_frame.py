#!/usr/bin/env python3

import sys

from src.carla.dataset.reader import load_frame
from src.carla.vis.open3d_viewer import show_frame


def main() -> None:
    if len(sys.argv) != 2:
        print("Użycie: python3 tools/dataset/show_dataset_frame.py detector_dataset_run_01/frame_000000")
        sys.exit(1)

    frame_dir = sys.argv[1]
    points, meta, ego_box = load_frame(frame_dir)

    print("[info] frame:", meta.get("frame"))
    print("[info] points shape:", points.shape)
    print("[info] num_objects:", len(meta.get("objects", [])))
    if "class_counts" in meta:
        print("[info] class_counts:", meta["class_counts"])

    show_frame(points, meta, ego_box)


if __name__ == "__main__":
    main()
