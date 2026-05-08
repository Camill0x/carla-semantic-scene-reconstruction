#!/usr/bin/env python3

import argparse
import shutil

from src.common.dataset import iter_frame_dirs, selected_run_dirs, train_val_test_split
from src.common.paths import repo_relative_or_absolute
from src.openpcdet.constants import CLASS_FILTERS
from src.openpcdet.infos import load_infos, write_infos
from src.openpcdet.paths import RAW_DATASET_ROOT, prepared_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenPCDet metadata for collected CARLA dataset runs")
    parser.add_argument(
        "--class-filter",
        choices=sorted(CLASS_FILTERS),
        default="carla_nuscenes6",
        help="Class/config filter",
    )
    parser.add_argument("--name", default="default", help="Prepared dataset variant name")
    run_selection = parser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument("--all", action="store_true", help="Use all run_XXXX directories under datasets/raw")
    run_selection.add_argument("--runs", nargs="+", metavar="RUN", help="Use selected raw run directories")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = RAW_DATASET_ROOT
    output_root = prepared_dataset_root(args.class_filter, args.name)
    class_names = CLASS_FILTERS[args.class_filter]

    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        if args.name == "default":
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Prepared dataset directory is not empty: {repo_relative_or_absolute(output_root)}")

    run_dirs = selected_run_dirs(source_root, None if args.all else args.runs)
    frame_dirs = iter_frame_dirs(run_dirs)
    print(f"Runs: [{', '.join(run_dir.name for run_dir in run_dirs)}]")
    print(f"Found {len(frame_dirs)} frame directories")

    infos = load_infos(frame_dirs, output_root, class_names)
    print(f"Loaded {len(infos)} valid samples")

    splits = train_val_test_split(
        items=infos,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_infos(output_root, splits)

    print(f"Train samples: {len(splits.train)}")
    print(f"Val samples: {len(splits.val)}")
    print(f"Test samples: {len(splits.test)}")
    print(f"Saved dataset: {repo_relative_or_absolute(output_root)}")


if __name__ == "__main__":
    main()
