#!/usr/bin/env python3

import argparse
import importlib.util
import shutil
from datetime import datetime
from pathlib import Path

from src.common.paths import repo_relative_or_absolute
from src.lanedet.artifacts import (
    copy_config,
    copy_log,
    copy_train_checkpoints,
    latest_lanedet_work_dir,
    recreate_dir,
    rel,
    write_json,
)
from src.lanedet.config import write_runtime_config
from src.lanedet.constants import LANEDET_PRESETS
from src.lanedet.metrics import build_tusimple_metrics
from src.lanedet.paths import (
    checkpoints_dir,
    lanedet_root,
    resolve_lanedet_cfg,
    run_dir,
)
from src.lanedet.runner import build_eval_command, build_train_command, run_lanedet_main


def generated_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LaneDet from the main project repo")
    parser.add_argument("--preset", default="laneatt-resnet34-tusimple", choices=sorted(LANEDET_PRESETS))
    parser.add_argument("--config", type=Path, help="Explicit LaneDet config file to use instead of --preset")
    parser.add_argument("--data-root", type=Path, help="Override dataset path from the config")
    parser.add_argument("--validate", action="store_true", help="Run LaneDet validation instead of training")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--load-from", type=Path, help="Checkpoint path to load; required with --validate")
    parser.add_argument("--finetune-from", type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", action="store_true")
    return parser.parse_args()


def load_config_module(config_path: Path):
    spec = importlib.util.spec_from_file_location("_lanedet_runtime_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_family_from_config(config_path: Path) -> str:
    module = load_config_module(config_path)
    try:
        dataset_type = module.dataset["train"]["type"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"Could not read dataset.train.type from {config_path}") from exc
    return str(dataset_type).lower()


def copy_common_outputs(source_work_dir: Path, output_dir: Path, gt_json: Path) -> dict:
    config_target = copy_config(source_work_dir, output_dir / "config.py")

    predictions_target = None
    predictions_source = source_work_dir / "tusimple_predictions.json"
    if predictions_source.exists():
        predictions_target = output_dir / "tusimple_predictions.json"
        shutil.copy2(predictions_source, predictions_target)
        metrics_target = output_dir / "metrics.json"
        write_json(metrics_target, build_tusimple_metrics(predictions_target, gt_json))
    else:
        metrics_target = None

    visualization_target = None
    visualization_source = source_work_dir / "visualization"
    if visualization_source.exists():
        visualization_target = output_dir / "visualization"
        shutil.copytree(visualization_source, visualization_target)

    return {
        "config": rel(config_target),
        "predictions": rel(predictions_target),
        "metrics": rel(metrics_target),
        "visualization": rel(visualization_target),
    }


def main() -> None:
    args = parse_args()
    if args.validate and args.load_from is None:
        raise ValueError("--load-from is required with --validate")
    if args.load_from is not None and not args.load_from.expanduser().resolve().exists():
        raise FileNotFoundError(args.load_from)

    cfg_file = resolve_lanedet_cfg(args.preset) if args.config is None else args.config.expanduser().resolve()
    dataset_root = args.data_root.expanduser().resolve() if args.data_root is not None else None
    if dataset_root is not None and not (dataset_root / "test_label.json").exists():
        raise FileNotFoundError(f"LaneDet dataset not found: {repo_relative_or_absolute(dataset_root)}")

    dataset_family = dataset_family_from_config(cfg_file)
    run_name = generated_run_name()
    output_dir = run_dir(run_name, dataset_family)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = output_dir / "_lanedet_work"
    recreate_dir(work_root)

    runtime_cfg = write_runtime_config(
        source_cfg=cfg_file,
        target_cfg=work_root / "input_config.py",
        lanedet_root=lanedet_root(),
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=args.workers,
    )

    command = (
        build_eval_command(runtime_cfg, args.load_from.expanduser().resolve(), work_root, args)
        if args.validate
        else build_train_command(runtime_cfg, work_root, args)
    )

    try:
        exit_code = run_lanedet_main(command)
        if exit_code != 0:
            raise SystemExit(exit_code)

        source_work_dir = latest_lanedet_work_dir(work_root)
        gt_json = (
            dataset_root / "test_label.json"
            if dataset_root is not None
            else Path(load_config_module(runtime_cfg).test_json_file)
        )
        common_outputs = copy_common_outputs(source_work_dir, output_dir, gt_json)

        if args.validate:
            log_target = copy_log(source_work_dir / "log.txt", output_dir / "test.log")
            checkpoint_outputs = {}
        else:
            log_target = copy_log(source_work_dir / "log.txt", output_dir / "train.log")
            checkpoint_outputs = {
                "checkpoints": {
                    key: rel(value)
                    for key, value in copy_train_checkpoints(source_work_dir, checkpoints_dir(run_name, dataset_family)).items()
                }
            }

        write_json(
            output_dir / "meta.json",
            {
                "mode": "validate" if args.validate else "train",
                "run": run_name,
                "dataset": dataset_family,
                "data_root_override": repo_relative_or_absolute(dataset_root) if dataset_root is not None else None,
                "preset": args.preset,
                "source_config": repo_relative_or_absolute(cfg_file),
                "load_from": repo_relative_or_absolute(args.load_from.expanduser().resolve()) if args.load_from is not None else None,
                "finetune_from": repo_relative_or_absolute(args.finetune_from.expanduser().resolve()) if args.finetune_from is not None else None,
                "log": rel(log_target),
                **common_outputs,
                **checkpoint_outputs,
            },
        )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    print(f"Results saved to: {repo_relative_or_absolute(output_dir)}")


if __name__ == "__main__":
    main()
