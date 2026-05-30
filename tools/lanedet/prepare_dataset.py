#!/usr/bin/env python3

import argparse
import shutil
from dataclasses import dataclass
from typing import List, Optional

from src.common.cli_logging import configure_logging
from src.common.dataset import iter_frame_dirs, selected_run_dirs, train_val_test_split
from src.common.paths import repo_relative_or_absolute
from src.lanedet.constants import LANEDET_DATASETS
from src.lanedet.datasets import load_samples, write_tusimple_dataset
from src.lanedet.paths import RAW_DATASET_ROOT, prepared_dataset_root


@dataclass(frozen=True)
class PrepareDatasetArgs:
    dataset_format: str
    name: str
    use_all: bool
    runs: Optional[List[str]]
    max_lanes: int
    line_width: int
    val_ratio: float
    test_ratio: float
    seed: int


def parse_args() -> PrepareDatasetArgs:
    """Parse command-line arguments for the LaneDet dataset preparation command."""
    parser = argparse.ArgumentParser(description="Build a TuSimple-like LaneDet dataset from collected CARLA runs")
    parser.add_argument("--format", choices=LANEDET_DATASETS, required=True, help="Prepared LaneDet dataset format")
    parser.add_argument("--name", default="default", help="Prepared dataset variant name")
    run_selection = parser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument("--all", action="store_true", help="Use all run_XXXX directories under datasets/raw")
    run_selection.add_argument("--runs", nargs="+", metavar="RUN", help="Use selected raw run directories")
    parser.add_argument("--max-lanes", type=int, default=5)
    parser.add_argument("--line-width", type=int, default=15, help="Segmentation mask lane line width in pixels")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parsed = parser.parse_args()
    return PrepareDatasetArgs(
        dataset_format=str(parsed.format),
        name=str(parsed.name),
        use_all=bool(parsed.all),
        runs=None if parsed.runs is None else [str(item) for item in parsed.runs],
        max_lanes=int(parsed.max_lanes),
        line_width=int(parsed.line_width),
        val_ratio=float(parsed.val_ratio),
        test_ratio=float(parsed.test_ratio),
        seed=int(parsed.seed),
    )


def main() -> None:
    """Run the LaneDet dataset preparation command."""
    args = parse_args()
    logger = configure_logging("tools.lanedet.prepare_dataset")
    source_root = RAW_DATASET_ROOT.resolve()
    output_root = prepared_dataset_root(args.name, args.dataset_format).resolve()

    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        if args.name == "default":
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Prepared dataset directory is not empty: {repo_relative_or_absolute(output_root)}")
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs = selected_run_dirs(source_root, None if args.use_all else args.runs)
    frame_dirs = iter_frame_dirs(run_dirs)

    logger.info("Preparing LaneDet dataset: %s", args.dataset_format)
    logger.info("Raw runs: [%s]", ", ".join(run_dir.name for run_dir in run_dirs))
    logger.info("Frame directories: %d", len(frame_dirs))
    logger.info("Output: %s", repo_relative_or_absolute(output_root))

    samples, stats = load_samples(frame_dirs, max_lanes=args.max_lanes, show_progress=True)

    logger.info("Skipped missing image/lanes files: %s", stats["skipped_missing_files"])
    logger.info("Skipped frames with no collected lane annotations: %s", stats["skipped_no_lanes_meta"])
    logger.info("Skipped frames without usable lane geometry: %s", stats["skipped_no_usable_lanes"])
    logger.info("Usable lane samples: %s", stats["usable_samples"])

    total_lanes = int(stats["total_lanes_in_samples"])
    avg_lanes_per_frame = (float(total_lanes) / float(len(samples))) if samples else 0.0
    logger.info("Total lanes in valid frames: %d (avg %.2f per frame)", total_lanes, avg_lanes_per_frame)

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

    logger.info("Train samples: %d", len(splits.train))
    logger.info("Val samples: %d", len(splits.val))
    logger.info("Test samples: %d", len(splits.test))
    logger.info("Saved dataset: %s", repo_relative_or_absolute(output_root))


if __name__ == "__main__":
    main()
