#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

from src.openpcdet.artifacts import resolve_checkpoint_selector
from src.openpcdet.commands import build_test_command, run_openpcdet_test
from src.openpcdet.common import (
    copy_eval_artifacts,
    default_eval_tag,
    make_checkpoint_link,
    recreate_dir,
    selected_epoch,
)
from src.openpcdet.constants import DEFAULT_DATASET_NAME, OPENPCDET_PRESETS
from src.openpcdet.paths import resolve_openpcdet_cfg, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an OpenPCDet checkpoint on the held-out test split")
    parser.add_argument("--preset", required=True, choices=sorted(OPENPCDET_PRESETS))
    parser.add_argument("--cfg-file", type=Path, help="Explicit OpenPCDet config file to use instead of --preset")
    parser.add_argument(
        "--checkpoint",
        default="last",
        help="Checkpoint selector: best, last, a filename under checkpoints/, an epoch number, or a path",
    )
    parser.add_argument("--name", required=True, help="Experiment name under results/openpcdet/")
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Prepared dataset name under datasets/openpcdet/carla_nuscenes6/",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-tag", help="Evaluation output tag under evaluations/. Defaults to a tag derived from --checkpoint")
    parser.add_argument("--set", dest="set_cfgs", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_file = resolve_openpcdet_cfg(args.preset) if args.cfg_file is None else args.cfg_file.expanduser().resolve()
    checkpoint = resolve_checkpoint_selector(args.name, args.checkpoint)
    epoch = selected_epoch(args, checkpoint)
    eval_tag = args.eval_tag or default_eval_tag(args.checkpoint, epoch)

    work_dir = run_dir(args.name) / "_openpcdet_test"
    recreate_dir(work_dir)
    ckpt_link_dir = work_dir / "checkpoint"
    ckpt_link_dir.mkdir(parents=True, exist_ok=True)

    try:
        eval_checkpoint = make_checkpoint_link(checkpoint, epoch, ckpt_link_dir)
        command = build_test_command(cfg_file, eval_checkpoint, work_dir, eval_tag, args)
        run_openpcdet_test(command)
        copy_eval_artifacts(work_dir, args.name, eval_tag, checkpoint)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
