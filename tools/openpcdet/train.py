#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.preset is not None:
        args.class_filter, cfg_file = resolve_openpcdet_preset(args.preset)
    else:
        cfg_file = args.cfg_file.expanduser().resolve()
        args.class_filter = cfg_file_class_filter(cfg_file)

    train_results_root = RESULTS_ROOT / "train" / args.class_filter
    args.name = generated_run_name(train_results_root)
    output_dir = train_results_root / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / "_openpcdet_train"
    recreate_dir(work_dir)

    command = build_train_command(cfg_file, work_dir, args)
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
        args=args,
        cfg_file=cfg_file,
        best_item=best_item,
        epochs_dir=epochs_dir,
        epoch_checkpoints=epoch_checkpoints,
    )
    write_json(output_dir / "meta.json", meta)
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Results saved to: {repo_relative_or_absolute(output_dir)}")


if __name__ == "__main__":
    main()
