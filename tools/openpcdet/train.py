#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

from src.openpcdet.artifacts import checkpoint_epoch, write_json
from src.openpcdet.commands import build_train_command, run_openpcdet_train
from src.openpcdet.common import (
    append_log_summary,
    build_training_summary,
    collect_validation_history,
    copy_config_snapshot,
    copy_latest_log,
    copy_selected_checkpoints,
    copy_tensorboard,
    print_artifact_summary,
    recreate_dir,
    training_final_lines,
)
from src.openpcdet.constants import DEFAULT_DATASET_NAME, OPENPCDET_PRESETS
from src.openpcdet.paths import (
    checkpoints_dir,
    config_dir,
    evaluations_dir,
    resolve_openpcdet_cfg,
    run_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an OpenPCDet model from the main project repo")
    parser.add_argument("--preset", required=True, choices=sorted(OPENPCDET_PRESETS))
    parser.add_argument("--cfg-file", type=Path, help="Explicit OpenPCDet config file to use instead of --preset")
    parser.add_argument("--pretrained-model", type=Path)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", required=True, help="Experiment name under results/openpcdet/")
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Prepared dataset name under datasets/openpcdet/carla_nuscenes6/",
    )
    parser.add_argument("--keep-all-ckpt", action="store_true", help="Keep per-epoch checkpoints after training")
    parser.add_argument("--best-metric", default="mAP", help="Metric used to choose the best checkpoint on val")
    parser.add_argument("--set", dest="set_cfgs", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_file = resolve_openpcdet_cfg(args.preset) if args.cfg_file is None else args.cfg_file.expanduser().resolve()
    output_dir = run_dir(args.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / "_openpcdet_train"
    recreate_dir(work_dir)

    command = build_train_command(cfg_file, work_dir, args)
    run_openpcdet_train(command)

    ckpt_work_dir = work_dir / "ckpt"
    epoch_checkpoints = sorted(ckpt_work_dir.glob("checkpoint_epoch_*.pth"), key=checkpoint_epoch)
    if not epoch_checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_work_dir}")

    eval_dir = evaluations_dir(args.name)
    ckpt_dir = checkpoints_dir(args.name)
    log_dir = output_dir / "logs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    val_history, best_item = collect_validation_history(work_dir, epoch_checkpoints, args.best_metric)
    best_target, last_target, epochs_dir, last_source = copy_selected_checkpoints(
        epoch_checkpoints,
        best_item,
        ckpt_dir,
        args.keep_all_ckpt,
    )

    copy_config_snapshot(cfg_file, work_dir, config_dir(args.name) / "used.yaml")
    copy_latest_log(work_dir, log_dir / "train.log")
    copy_tensorboard(work_dir, output_dir / "tensorboard")

    write_json(eval_dir / "val_history.json", val_history)
    write_json(eval_dir / "best_val.json", best_item)

    summary_json_path = output_dir / "summary.json"
    summary = build_training_summary(
        args=args,
        cfg_file=cfg_file,
        best_item=best_item,
        last_source=last_source,
        best_target=best_target,
        last_target=last_target,
        epochs_dir=epochs_dir,
        epoch_checkpoints=epoch_checkpoints,
        output_dir=output_dir,
        eval_dir=eval_dir,
        log_dir=log_dir,
    )
    write_json(summary_json_path, summary)

    final_lines = training_final_lines(
        best_target,
        last_target,
        eval_dir,
        summary_json_path,
        output_dir / "tensorboard",
    )
    append_log_summary(log_dir / "train.log", final_lines)
    print_artifact_summary(final_lines)
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
