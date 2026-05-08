#!/usr/bin/env python3

import argparse
import shutil

from src.common.dataset import iter_frame_dirs, selected_run_dirs, train_val_test_split
from src.common.paths import repo_relative_or_absolute
from src.lanedet.constants import LANEDET_DATASETS
from src.lanedet.datasets import load_samples, write_tusimple_dataset
from src.lanedet.paths import RAW_DATASET_ROOT, prepared_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TuSimple-like LaneDet dataset from collected CARLA runs")
    parser.add_argument("--format", choices=LANEDET_DATASETS, required=True, help="Prepared LaneDet dataset format")
    parser.add_argument("--name", default="default", help="Prepared dataset variant name")
    run_selection = parser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument("--all", action="store_true", help="Use all run_XXXX directories under datasets/raw")
    run_selection.add_argument("--runs", nargs="+", metavar="RUN", help="Use selected raw run directories")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lanes", type=int, default=5)
    parser.add_argument("--line-width", type=int, default=15, help="Segmentation mask lane line width in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = RAW_DATASET_ROOT.resolve()
    output_root = prepared_dataset_root(args.name, args.format).resolve()

    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        if args.name == "default":
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Prepared dataset directory is not empty: {repo_relative_or_absolute(output_root)}")
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs = selected_run_dirs(source_root, None if args.all else args.runs)
    frame_dirs = iter_frame_dirs(run_dirs)

    print(f"Preparing LaneDet dataset: {args.format}")
    print(f"Raw runs: [{', '.join(run_dir.name for run_dir in run_dirs)}]")
    print(f"Frame directories: {len(frame_dirs)}")
    print(f"Output: {repo_relative_or_absolute(output_root)}")

    samples, stats = load_samples(frame_dirs, max_lanes=args.max_lanes, show_progress=True)
    print(f"Loaded {len(samples)} valid lane samples")

    splits = train_val_test_split(
        items=samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_tusimple_dataset(
        output_root=output_root,
        splits=splits,
        line_width=args.line_width,
    )

    print(f"Skipped missing image/lanes files: {stats['skipped_missing_files']}")
    print(f"Skipped frames with no collected lane annotations: {stats['skipped_no_lanes_meta']}")
    print(f"Skipped frames without usable lane geometry: {stats['skipped_no_usable_lanes']}")
    print(f"Usable lane samples: {stats['usable_samples']}")
    print(f"Train samples: {len(splits.train)}")
    print(f"Val samples: {len(splits.val)}")
    print(f"Test samples: {len(splits.test)}")
    print(f"Saved dataset: {repo_relative_or_absolute(output_root)}")


if __name__ == "__main__":
    main()
