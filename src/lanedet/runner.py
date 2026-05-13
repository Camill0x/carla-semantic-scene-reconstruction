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


def build_train_command(
    cfg_file: Path,
    work_dir: Path,
    *,
    gpus: Sequence[int],
    seed: int,
    load_from: Optional[Path],
    finetune_from: Optional[Path],
    view: bool,
) -> List[str]:
    command = [
        str(cfg_file),
        "--work_dirs",
        str(work_dir),
        "--gpus",
        *[str(gpu) for gpu in gpus],
        "--seed",
        str(seed),
    ]
    if load_from is not None:
        command.extend(["--load_from", str(load_from.expanduser().resolve())])
    if finetune_from is not None:
        command.extend(["--finetune_from", str(finetune_from.expanduser().resolve())])
    if view:
        command.append("--view")
    command.append("--quiet_log")
    return command


def build_eval_command(
    cfg_file: Path,
    checkpoint: Path,
    work_dir: Path,
    *,
    gpus: Sequence[int],
    seed: int,
    view: bool,
) -> List[str]:
    command = [
        str(cfg_file),
        "--validate",
        "--load_from",
        str(checkpoint),
        "--work_dirs",
        str(work_dir),
        "--gpus",
        *[str(gpu) for gpu in gpus],
        "--seed",
        str(seed),
    ]
    if view:
        command.append("--view")
    command.append("--quiet_log")
    return command


def resolve_run_config(preset: Optional[str], config_path: Optional[Path]) -> Tuple[Path, str]:
    if preset is not None:
        return resolve_lanedet_cfg(preset), preset_model(preset)

    assert config_path is not None
    cfg_file = config_path.expanduser().resolve()
    return cfg_file, model_from_config_path(cfg_file)


def resolve_data_root(data_root: Optional[Path], preset: Optional[str]) -> Optional[Path]:
    if data_root is not None:
        return data_root.expanduser().resolve()
    if preset is not None:
        return Path(prepared_dataset_root(dataset=preset_dataset(preset))).resolve()
    return None


def validate_run_args(validate: bool, load_from: Optional[Path]) -> None:
    if validate and load_from is None:
        raise ValueError("--load-from is required with --validate")
    if load_from is not None and not load_from.expanduser().resolve().exists():
        raise FileNotFoundError(load_from)
