import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional, Sequence

from src.common.paths import repo_relative_or_absolute
from src.openpcdet.artifacts import checkpoint_epoch, latest_file, read_json, write_json
from src.openpcdet.paths import relative_to_openpcdet


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_latest_log(work_dir: Path, target: Path) -> None:
    source = latest_file(work_dir.glob("train_*.log"))
    if source is None:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_config_snapshot(cfg_file: Path, work_dir: Path, target: Path) -> None:
    source = work_dir / Path(relative_to_openpcdet(cfg_file)).name
    if not source.exists():
        source = cfg_file
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tensorboard(work_dir: Path, target: Path) -> None:
    source = work_dir / "tensorboard"
    if not source.exists():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def read_train_eval_metrics(work_dir: Path, epoch: int) -> Dict:
    result_root = work_dir / "eval" / "eval_with_train" / f"epoch_{epoch}"
    candidates = sorted(result_root.glob("*/metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No training-eval metrics found for epoch {epoch} under {result_root}")
    return read_json(candidates[0])


def select_best_validation_checkpoint(
    work_dir: Path,
    checkpoints: Sequence[Path],
    best_metric: str,
) -> Dict:
    best_item = None
    best_value = float("-inf")

    for checkpoint in checkpoints:
        epoch = checkpoint_epoch(checkpoint)
        metrics_payload = read_train_eval_metrics(work_dir, epoch)
        metrics = metrics_payload.get("metrics", {})
        metric_value = float(metrics.get(best_metric, float("-inf")))
        item = {
            "epoch": epoch,
            "checkpoint_ref": f"epoch_{epoch}",
            "metric_name": best_metric,
            "metric_value": metric_value,
            "metrics": metrics,
        }
        if best_item is None or metric_value > best_value:
            best_value = metric_value
            best_item = item

    if best_item is None:
        raise RuntimeError("Validation did not produce a best checkpoint")
    return best_item


def copy_selected_checkpoints(
    checkpoints: Sequence[Path],
    best_item: Dict,
    target_dir: Path,
    keep_all: bool,
) -> Optional[Path]:
    best_source = next(path for path in checkpoints if checkpoint_epoch(path) == best_item["epoch"])
    last_source = checkpoints[-1]
    shutil.copy2(best_source, target_dir / "best.ckpt")
    shutil.copy2(last_source, target_dir / "last.ckpt")

    epochs_dir = None
    if keep_all:
        epochs_dir = target_dir / "epochs"
        recreate_dir(epochs_dir)
        for checkpoint in checkpoints:
            shutil.copy2(checkpoint, epochs_dir / f"epoch_{checkpoint_epoch(checkpoint):03d}.ckpt")

    return epochs_dir


def build_train_meta(
    *,
    args: Namespace,
    cfg_file: Path,
    best_item: Dict,
    epochs_dir: Optional[Path],
    epoch_checkpoints: Sequence[Path],
) -> Dict:
    return {
        "mode": "train",
        "run": args.name,
        "dataset": {
            "family": args.class_filter,
            "name": args.dataset_name,
        },
        "config": {
            "preset": args.preset,
            "source": repo_relative_or_absolute(cfg_file),
            "overrides": args.set_cfgs or [],
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "checkpoint_interval": 1,
            "keep_all_ckpt": args.keep_all_ckpt,
            "pretrained_model": (
                repo_relative_or_absolute(args.pretrained_model) if args.pretrained_model is not None else None
            ),
            "resume_checkpoint": repo_relative_or_absolute(args.ckpt) if args.ckpt is not None else None,
        },
        "validation": {
            "split": "val",
            "best_metric": args.best_metric,
            "best_epoch": best_item["epoch"],
            "best_value": best_item["metric_value"],
        },
        "checkpoints": {
            "best_epoch": best_item["epoch"],
            "last_epoch": checkpoint_epoch(epoch_checkpoints[-1]),
            "kept_epoch_count": len(epoch_checkpoints) if epochs_dir is not None else 0,
        },
    }


def build_test_meta(*, args: Namespace, cfg_file: Path, checkpoint: Path) -> Dict:
    epoch = checkpoint_epoch(checkpoint)
    return {
        "mode": "test",
        "run": args.name,
        "dataset": {
            "family": args.class_filter,
            "name": args.dataset_name,
            "split": "test",
        },
        "config": {
            "preset": args.preset,
            "source": repo_relative_or_absolute(cfg_file),
            "overrides": args.set_cfgs or [],
        },
        "checkpoint": {
            "path": repo_relative_or_absolute(checkpoint),
            "epoch": epoch if epoch >= 0 else None,
        },
        "evaluation": {
            "batch_size": args.batch_size,
            "workers": args.workers,
        },
    }


def make_checkpoint_link(checkpoint: Path, epoch: str, tmp_dir: Path) -> Path:
    name = "checkpoint_no_number.pth" if epoch == "no_number" else f"checkpoint_epoch_{epoch}.pth"
    link_path = tmp_dir / name
    os.symlink(checkpoint, link_path)
    return link_path


def copy_test_artifacts(work_dir: Path, output_dir: Path, checkpoint: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = latest_file(work_dir.glob("eval/**/test/metrics.json"))
    result_path = latest_file(work_dir.glob("eval/**/test/result.pkl"))
    log_path = latest_file(work_dir.glob("eval/**/test/log_eval_*.txt"))

    if metrics_path is None:
        raise FileNotFoundError(f"No metrics.json produced under {work_dir}")

    metrics = read_json(metrics_path)
    metrics["checkpoint"] = repo_relative_or_absolute(checkpoint)
    write_json(output_dir / "metrics.json", metrics)
    if result_path is not None:
        shutil.copy2(result_path, output_dir / "result.pkl")
    if log_path is not None:
        shutil.copy2(log_path, output_dir / "test.log")
