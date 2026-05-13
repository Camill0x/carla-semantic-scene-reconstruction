#!/usr/bin/env python3

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.common.cli_logging import configure_logging
from src.common.paths import repo_relative_or_absolute
from src.lanedet.artifacts import (
    copy_common_outputs,
    copy_log,
    copy_train_checkpoints,
    latest_lanedet_work_dir,
    recreate_dir,
    write_run_metadata,
)
from src.lanedet.config import dataset_family_from_config, load_config_module, write_runtime_config
from src.lanedet.constants import LANEDET_PRESETS
from src.lanedet.paths import LANEDET_ROOT, checkpoints_dir, generated_run_name, run_dir
from src.lanedet.runner import (
    build_eval_command,
    build_train_command,
    resolve_data_root,
    resolve_run_config,
    run_lanedet_main,
    validate_run_args,
)


@dataclass(frozen=True)
class LaneDetArgs:
    preset: Optional[str]
    config: Optional[Path]
    data_root: Optional[Path]
    validate: bool
    batch_size: Optional[int]
    epochs: Optional[int]
    workers: Optional[int]
    load_from: Optional[Path]
    finetune_from: Optional[Path]
    gpus: List[int]
    seed: int
    view: bool


def parse_args() -> LaneDetArgs:
    """Parse command-line arguments for the LaneDet project wrapper command."""
    parser = argparse.ArgumentParser(description="Run LaneDet from the main project repo")
    config_selection = parser.add_mutually_exclusive_group(required=True)
    config_selection.add_argument("--preset", choices=sorted(LANEDET_PRESETS))
    config_selection.add_argument("--config", type=Path, help="Explicit LaneDet config file to use instead of --preset")
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
    parsed = parser.parse_args()
    return LaneDetArgs(
        preset=None if parsed.preset is None else str(parsed.preset),
        config=parsed.config,
        data_root=parsed.data_root,
        validate=bool(parsed.validate),
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        workers=parsed.workers,
        load_from=parsed.load_from,
        finetune_from=parsed.finetune_from,
        gpus=[int(gpu) for gpu in parsed.gpus],
        seed=int(parsed.seed),
        view=bool(parsed.view),
    )


def main() -> None:
    """Run the LaneDet project wrapper command."""
    args = parse_args()
    logger = configure_logging("tools.lanedet.main")
    validate_run_args(args.validate, args.load_from)

    cfg_file, model = resolve_run_config(args.preset, args.config)
    mode = "test" if args.validate else "train"
    dataset_root = resolve_data_root(args.data_root, args.preset)
    if dataset_root is not None and not (dataset_root / "test_label.json").exists():
        raise FileNotFoundError(f"LaneDet dataset not found: {repo_relative_or_absolute(dataset_root)}")

    dataset_family = dataset_family_from_config(cfg_file)
    run_name = generated_run_name()
    output_dir = run_dir(run_name, mode, model, dataset_family)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = output_dir / "_lanedet_work"
    recreate_dir(work_root)

    runtime_cfg = write_runtime_config(
        source_cfg=cfg_file,
        target_cfg=work_root / "input_config.py",
        lanedet_root=LANEDET_ROOT,
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=args.workers,
    )

    eval_checkpoint = None if args.load_from is None else args.load_from.expanduser().resolve()
    if args.validate:
        assert eval_checkpoint is not None
        command = build_eval_command(
            runtime_cfg,
            eval_checkpoint,
            work_root,
            gpus=args.gpus,
            seed=args.seed,
            view=args.view,
        )
    else:
        command = build_train_command(
            runtime_cfg,
            work_root,
            gpus=args.gpus,
            seed=args.seed,
            load_from=args.load_from,
            finetune_from=args.finetune_from,
            view=args.view,
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
        copy_common_outputs(source_work_dir, output_dir, gt_json)

        if args.validate:
            copy_log(source_work_dir / "log.txt", output_dir / "test.log")
        else:
            copy_log(source_work_dir / "log.txt", output_dir / "train.log")
            copy_train_checkpoints(source_work_dir, checkpoints_dir(run_name, mode, model, dataset_family))

        write_run_metadata(
            output_dir=output_dir,
            mode=mode,
            run_name=run_name,
            dataset=dataset_family,
            model=model,
            data_root=dataset_root,
            preset=args.preset,
            source_config=cfg_file,
            load_from=args.load_from.expanduser().resolve() if args.load_from is not None else None,
            finetune_from=args.finetune_from.expanduser().resolve() if args.finetune_from is not None else None,
        )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    logger.info("Results saved to: %s", repo_relative_or_absolute(output_dir))


if __name__ == "__main__":
    main()
