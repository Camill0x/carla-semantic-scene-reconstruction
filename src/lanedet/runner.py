import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from src.common.paths import repo_root
from src.lanedet.paths import (
    LANEDET_ROOT,
    model_from_config_path,
    prepared_dataset_root,
    preset_dataset,
    preset_model,
    resolve_lanedet_cfg,
)


def run_lanedet_main(args: Sequence[str]) -> int:
    command = [sys.executable, str(LANEDET_ROOT / "main.py"), *args]
    completed = subprocess.run(command, cwd=repo_root(), check=False)
    return int(completed.returncode)


def build_train_command(cfg_file: Path, work_dir: Path, args) -> List[str]:
    command = [
        str(cfg_file),
        "--work_dirs",
        str(work_dir),
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--seed",
        str(args.seed),
    ]
    if args.load_from is not None:
        command.extend(["--load_from", str(args.load_from.expanduser().resolve())])
    if args.finetune_from is not None:
        command.extend(["--finetune_from", str(args.finetune_from.expanduser().resolve())])
    if args.view:
        command.append("--view")
    command.append("--quiet_log")
    return command


def build_eval_command(cfg_file: Path, checkpoint: Path, work_dir: Path, args) -> List[str]:
    command = [
        str(cfg_file),
        "--validate",
        "--load_from",
        str(checkpoint),
        "--work_dirs",
        str(work_dir),
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--seed",
        str(args.seed),
    ]
    if args.view:
        command.append("--view")
    command.append("--quiet_log")
    return command


def resolve_run_config(args) -> Tuple[Path, str]:
    if args.preset is not None:
        return resolve_lanedet_cfg(args.preset), preset_model(args.preset)

    cfg_file = args.config.expanduser().resolve()
    return cfg_file, model_from_config_path(cfg_file)


def resolve_data_root(args) -> Optional[Path]:
    if args.data_root is not None:
        return args.data_root.expanduser().resolve()
    if args.preset is not None:
        return prepared_dataset_root(dataset=preset_dataset(args.preset)).resolve()
    return None


def validate_run_args(args) -> None:
    if args.validate and args.load_from is None:
        raise ValueError("--load-from is required with --validate")
    if args.load_from is not None and not args.load_from.expanduser().resolve().exists():
        raise FileNotFoundError(args.load_from)
