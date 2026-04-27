import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

from src.common.paths import repo_root
from src.lanedet.paths import lanedet_root


def run_lanedet_main(args: Sequence[str]) -> int:
    command = [sys.executable, str(lanedet_root() / "main.py"), *args]
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
