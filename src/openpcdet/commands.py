from pathlib import Path
from typing import List, Optional, Sequence

from src.openpcdet.paths import prepared_dataset_data_path, relative_to_openpcdet
from src.openpcdet.runner import extend_with_set_args, run_openpcdet_tool


def build_train_command(
    cfg_file: Path,
    work_dir: Path,
    *,
    workers: int,
    run_name: str,
    pretrained_model: Optional[Path],
    resume_checkpoint: Optional[Path],
    batch_size: Optional[int],
    epochs: Optional[int],
    class_filter: str,
    dataset_name: str,
    set_cfgs: Optional[Sequence[str]],
) -> List[str]:
    command = [
        "--cfg_file",
        relative_to_openpcdet(cfg_file),
        "--workers",
        str(workers),
        "--extra_tag",
        run_name,
        "--max_waiting_mins",
        "0",
        "--max_ckpt_save_num",
        "100000",
        "--num_epochs_to_eval",
        "100000",
        "--wo_gpu_stat",
        "--quiet_config",
        "--output_dir",
        str(work_dir),
    ]
    if pretrained_model is not None:
        command.extend(["--pretrained_model", str(pretrained_model.expanduser().resolve())])
    if resume_checkpoint is not None:
        command.extend(["--ckpt", str(resume_checkpoint.expanduser().resolve())])
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])

    cfg_overrides = ["DATA_CONFIG.DATA_PATH", prepared_dataset_data_path(class_filter, dataset_name)]
    if set_cfgs:
        cfg_overrides.extend(set_cfgs)
    return extend_with_set_args(command, cfg_overrides)


def build_test_command(
    cfg_file: Path,
    checkpoint: Path,
    work_dir: Path,
    eval_tag: str,
    *,
    workers: int,
    run_name: str,
    batch_size: Optional[int],
    class_filter: str,
    dataset_name: str,
    set_cfgs: Optional[Sequence[str]],
) -> List[str]:
    command = [
        "--cfg_file",
        relative_to_openpcdet(cfg_file),
        "--ckpt",
        str(checkpoint),
        "--workers",
        str(workers),
        "--extra_tag",
        run_name,
        "--eval_tag",
        eval_tag,
        "--output_dir",
        str(work_dir),
        "--quiet_config",
    ]
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])

    cfg_overrides = [
        "DATA_CONFIG.DATA_PATH",
        prepared_dataset_data_path(class_filter, dataset_name),
        "DATA_CONFIG.DATA_SPLIT.test",
        "test",
        "DATA_CONFIG.INFO_PATH.test",
        "['infos/infos_test.pkl']",
    ]
    if set_cfgs:
        cfg_overrides.extend(set_cfgs)
    return extend_with_set_args(command, cfg_overrides)


def run_openpcdet_train(command: List[str]) -> None:
    exit_code = run_openpcdet_tool("train.py", command)
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_openpcdet_test(command: List[str]) -> None:
    exit_code = run_openpcdet_tool("test.py", command)
    if exit_code != 0:
        raise SystemExit(exit_code)
