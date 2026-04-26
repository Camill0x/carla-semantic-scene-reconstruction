from argparse import Namespace
from pathlib import Path
from typing import List

from src.openpcdet.paths import prepared_dataset_data_path, relative_to_openpcdet
from src.openpcdet.runner import extend_with_set_args, run_openpcdet_tool


def build_train_command(cfg_file: Path, work_dir: Path, args: Namespace) -> List[str]:
    command = [
        "--cfg_file", relative_to_openpcdet(cfg_file),
        "--workers", str(args.workers),
        "--extra_tag", args.name,
        "--max_waiting_mins", "0",
        "--max_ckpt_save_num", "100000",
        "--num_epochs_to_eval", "100000",
        "--wo_gpu_stat",
        "--quiet_config",
        "--output_dir", str(work_dir),
    ]
    if args.pretrained_model is not None:
        command.extend(["--pretrained_model", str(args.pretrained_model.expanduser().resolve())])
    if args.ckpt is not None:
        command.extend(["--ckpt", str(args.ckpt.expanduser().resolve())])
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])

    set_cfgs = ["DATA_CONFIG.DATA_PATH", prepared_dataset_data_path(args.dataset_name)]
    if args.set_cfgs:
        set_cfgs.extend(args.set_cfgs)
    return extend_with_set_args(command, set_cfgs)


def build_test_command(cfg_file: Path, checkpoint: Path, work_dir: Path, eval_tag: str, args: Namespace) -> List[str]:
    command = [
        "--cfg_file", relative_to_openpcdet(cfg_file),
        "--ckpt", str(checkpoint),
        "--workers", str(args.workers),
        "--extra_tag", args.name,
        "--eval_tag", eval_tag,
        "--output_dir", str(work_dir),
        "--quiet_config",
    ]
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])

    set_cfgs = [
        "DATA_CONFIG.DATA_PATH", prepared_dataset_data_path(args.dataset_name),
        "DATA_CONFIG.DATA_SPLIT.test", "test",
        "DATA_CONFIG.INFO_PATH.test", "['infos/infos_test.pkl']",
    ]
    if args.set_cfgs:
        set_cfgs.extend(args.set_cfgs)
    return extend_with_set_args(command, set_cfgs)


def run_openpcdet_train(command: List[str]) -> None:
    exit_code = run_openpcdet_tool("train.py", command)
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_openpcdet_test(command: List[str]) -> None:
    exit_code = run_openpcdet_tool("test.py", command)
    if exit_code != 0:
        raise SystemExit(exit_code)
