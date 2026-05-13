#!/usr/bin/env python3

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.common.cli_logging import configure_logging
from src.common.paths import repo_relative_or_absolute
from src.openpcdet.artifacts import checkpoint_epoch, write_json
from src.openpcdet.commands import build_test_command, run_openpcdet_test
from src.openpcdet.common import (
    build_test_meta,
    copy_config_snapshot,
    copy_test_artifacts,
    make_checkpoint_link,
    recreate_dir,
)
from src.openpcdet.constants import OPENPCDET_PRESETS
from src.openpcdet.paths import (
    RESULTS_ROOT,
    cfg_file_class_filter,
    generated_run_name,
    resolve_openpcdet_preset,
)


@dataclass(frozen=True)
class TestArgs:
    dataset_name: str
    preset: Optional[str]
    cfg_file: Optional[Path]
    ckpt: Path
    batch_size: Optional[int]
    workers: int
    set_cfgs: Optional[List[str]]


def parse_args() -> TestArgs:
    """Parse command-line arguments for the OpenPCDet evaluation command."""
    parser = argparse.ArgumentParser(description="Evaluate an OpenPCDet checkpoint on the held-out test split")
    parser.add_argument("--dataset-name", default="default", help="Prepared dataset variant name")
    cfg_source = parser.add_mutually_exclusive_group(required=True)
    cfg_source.add_argument("--preset", choices=sorted(OPENPCDET_PRESETS), help="Project config preset")
    cfg_source.add_argument("--cfg-file", type=Path, help="OpenPCDet config path")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--set", dest="set_cfgs", nargs=argparse.REMAINDER, help="Extra OpenPCDet config overrides")
    parsed = parser.parse_args()
    return TestArgs(
        dataset_name=str(parsed.dataset_name),
        preset=None if parsed.preset is None else str(parsed.preset),
        cfg_file=parsed.cfg_file,
        ckpt=parsed.ckpt,
        batch_size=parsed.batch_size,
        workers=int(parsed.workers),
        set_cfgs=None if parsed.set_cfgs is None else [str(item) for item in parsed.set_cfgs],
    )


def main() -> None:
    """Run the OpenPCDet evaluation command."""
    args = parse_args()
    logger = configure_logging("tools.openpcdet.test")

    if args.preset is not None:
        class_filter, cfg_file = resolve_openpcdet_preset(args.preset)
    else:
        assert args.cfg_file is not None
        cfg_file = args.cfg_file.expanduser().resolve()
        class_filter = cfg_file_class_filter(cfg_file)

    test_results_root = RESULTS_ROOT / "test" / class_filter
    run_name = generated_run_name(test_results_root)

    checkpoint = args.ckpt.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    output_dir = test_results_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_openpcdet_test"
    recreate_dir(work_dir)
    ckpt_link_dir = work_dir / "checkpoint"
    ckpt_link_dir.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint_epoch_id = checkpoint_epoch(checkpoint)
        epoch = str(checkpoint_epoch_id) if checkpoint_epoch_id >= 0 else "no_number"
        eval_checkpoint = make_checkpoint_link(checkpoint, epoch, ckpt_link_dir)

        command = build_test_command(
            cfg_file,
            eval_checkpoint,
            work_dir,
            "test",
            workers=args.workers,
            run_name=run_name,
            batch_size=args.batch_size,
            class_filter=class_filter,
            dataset_name=args.dataset_name,
            set_cfgs=args.set_cfgs,
        )
        run_openpcdet_test(command)

        copy_config_snapshot(cfg_file, work_dir, output_dir / "source_config.yaml")
        copy_test_artifacts(work_dir, output_dir, checkpoint)

        meta = build_test_meta(
            run_name=run_name,
            class_filter=class_filter,
            dataset_name=args.dataset_name,
            preset=args.preset,
            set_cfgs=args.set_cfgs,
            batch_size=args.batch_size,
            workers=args.workers,
            cfg_file=cfg_file,
            checkpoint=checkpoint,
        )
        write_json(output_dir / "meta.json", meta)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("Results saved to: %s", repo_relative_or_absolute(output_dir))


if __name__ == "__main__":
    main()
