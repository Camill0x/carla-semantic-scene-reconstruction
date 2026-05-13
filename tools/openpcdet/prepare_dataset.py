#!/usr/bin/env python3

import argparse
import shutil
from dataclasses import dataclass
from typing import List, Optional

from src.common.cli_logging import configure_logging
from src.common.dataset import iter_frame_dirs, selected_run_dirs, train_val_test_split
from src.common.paths import repo_relative_or_absolute
from src.openpcdet.constants import CLASS_FILTERS
from src.openpcdet.infos import load_infos, write_infos
from src.openpcdet.paths import RAW_DATASET_ROOT, prepared_dataset_root


@dataclass(frozen=True)
class PrepareDatasetArgs:
    class_filter: str
    name: str
    use_all: bool
    runs: Optional[List[str]]
    val_ratio: float
    test_ratio: float
    seed: int


def parse_args() -> PrepareDatasetArgs:
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
    parsed = parser.parse_args()
    return PrepareDatasetArgs(
        class_filter=str(parsed.class_filter),
        name=str(parsed.name),
        use_all=bool(parsed.all),
        runs=None if parsed.runs is None else [str(item) for item in parsed.runs],
        val_ratio=float(parsed.val_ratio),
        test_ratio=float(parsed.test_ratio),
        seed=int(parsed.seed),
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging("tools.openpcdet.prepare_dataset")
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

    run_dirs = selected_run_dirs(source_root, None if args.use_all else args.runs)
    frame_dirs = iter_frame_dirs(run_dirs)
    logger.info("runs: [%s]", ", ".join(run_dir.name for run_dir in run_dirs))
    logger.info("found %d frame directories", len(frame_dirs))

    infos = load_infos(frame_dirs, output_root, class_names)
    logger.info("loaded %d valid samples", len(infos))

    splits = train_val_test_split(
        items=infos,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_infos(output_root, splits)

    logger.info("train samples: %d", len(splits.train))
    logger.info("val samples: %d", len(splits.val))
    logger.info("test samples: %d", len(splits.test))
    logger.info("saved dataset: %s", repo_relative_or_absolute(output_root))


if __name__ == "__main__":
    main()
