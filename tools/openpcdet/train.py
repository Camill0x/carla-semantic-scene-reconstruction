#!/usr/bin/env python3

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.common.cli_logging import configure_logging
from src.common.paths import repo_relative_or_absolute
from src.openpcdet.artifacts import checkpoint_epoch, write_json
from src.openpcdet.commands import build_train_command, run_openpcdet_train
from src.openpcdet.common import (
    build_train_meta,
    copy_config_snapshot,
    copy_latest_log,
    copy_selected_checkpoints,
    copy_tensorboard,
    recreate_dir,
    select_best_validation_checkpoint,
)
from src.openpcdet.constants import OPENPCDET_PRESETS
from src.openpcdet.paths import (
    RESULTS_ROOT,
    cfg_file_class_filter,
    generated_run_name,
    resolve_openpcdet_preset,
)


@dataclass(frozen=True)
class TrainArgs:
    dataset_name: str
    preset: Optional[str]
    cfg_file: Optional[Path]
    pretrained_model: Optional[Path]
    ckpt: Optional[Path]
    batch_size: Optional[int]
    epochs: Optional[int]
    workers: int
    keep_all_ckpt: bool
    best_metric: str
    set_cfgs: Optional[List[str]]


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train an OpenPCDet model from the main project repo")
    parser.add_argument("--dataset-name", default="default", help="Prepared dataset variant name")
    cfg_source = parser.add_mutually_exclusive_group(required=True)
    cfg_source.add_argument("--preset", choices=sorted(OPENPCDET_PRESETS), help="Project config preset")
    cfg_source.add_argument("--cfg-file", type=Path, help="OpenPCDet config path")
    parser.add_argument("--pretrained-model", type=Path)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-all-ckpt", action="store_true", help="Keep per-epoch checkpoints after training")
    parser.add_argument("--best-metric", default="mAP", help="Metric used to choose the best checkpoint on val")
    parser.add_argument("--set", dest="set_cfgs", nargs=argparse.REMAINDER, help="Extra OpenPCDet config overrides")
    parsed = parser.parse_args()
    return TrainArgs(
        dataset_name=str(parsed.dataset_name),
        preset=None if parsed.preset is None else str(parsed.preset),
        cfg_file=parsed.cfg_file,
        pretrained_model=parsed.pretrained_model,
        ckpt=parsed.ckpt,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        workers=int(parsed.workers),
        keep_all_ckpt=bool(parsed.keep_all_ckpt),
        best_metric=str(parsed.best_metric),
        set_cfgs=None if parsed.set_cfgs is None else [str(item) for item in parsed.set_cfgs],
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging("tools.openpcdet.train")

    if args.preset is not None:
        class_filter, cfg_file = resolve_openpcdet_preset(args.preset)
    else:
        assert args.cfg_file is not None
        cfg_file = args.cfg_file.expanduser().resolve()
        class_filter = cfg_file_class_filter(cfg_file)

    train_results_root = RESULTS_ROOT / "train" / class_filter
    run_name = generated_run_name(train_results_root)
    output_dir = train_results_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / "_openpcdet_train"
    recreate_dir(work_dir)

    command = build_train_command(
        cfg_file,
        work_dir,
        workers=args.workers,
        run_name=run_name,
        pretrained_model=args.pretrained_model,
        resume_checkpoint=args.ckpt,
        batch_size=args.batch_size,
        epochs=args.epochs,
        class_filter=class_filter,
        dataset_name=args.dataset_name,
        set_cfgs=args.set_cfgs,
    )
    run_openpcdet_train(command)

    ckpt_work_dir = work_dir / "ckpt"
    epoch_checkpoints = sorted(ckpt_work_dir.glob("checkpoint_epoch_*.pth"), key=checkpoint_epoch)
    if not epoch_checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_work_dir}")

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_item = select_best_validation_checkpoint(work_dir, epoch_checkpoints, args.best_metric)
    epochs_dir = copy_selected_checkpoints(
        epoch_checkpoints,
        best_item,
        ckpt_dir,
        args.keep_all_ckpt,
    )

    copy_config_snapshot(cfg_file, work_dir, output_dir / "source_config.yaml")
    copy_latest_log(work_dir, output_dir / "train.log")
    copy_tensorboard(work_dir, output_dir / "tensorboard")

    write_json(output_dir / "metrics.json", best_item)

    meta = build_train_meta(
        run_name=run_name,
        class_filter=class_filter,
        dataset_name=args.dataset_name,
        preset=args.preset,
        cfg_file=cfg_file,
        set_cfgs=args.set_cfgs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        keep_all_ckpt=args.keep_all_ckpt,
        pretrained_model=args.pretrained_model,
        resume_checkpoint=args.ckpt,
        best_metric=args.best_metric,
        best_item=best_item,
        epochs_dir=epochs_dir,
        epoch_checkpoints=epoch_checkpoints,
    )
    write_json(output_dir / "meta.json", meta)
    shutil.rmtree(work_dir, ignore_errors=True)
    logger.info("results saved to: %s", repo_relative_or_absolute(output_dir))


if __name__ == "__main__":
    main()
