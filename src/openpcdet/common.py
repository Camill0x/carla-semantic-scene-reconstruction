import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

from src.common.paths import repo_relative_or_absolute
from src.common.typing_aliases import JsonDict
from src.openpcdet.artifacts import checkpoint_epoch, latest_file, read_json, write_json
from src.openpcdet.paths import relative_to_openpcdet


def recreate_dir(path: Path) -> None:
    """Remove a directory if it exists and recreate it empty."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_latest_log(work_dir: Path, target: Path) -> None:
    """Copy the newest OpenPCDet log file into the target location."""
    source = latest_file(work_dir.glob("train_*.log"))
    if source is None:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_config_snapshot(cfg_file: Path, work_dir: Path, target: Path) -> None:
    """Copy the effective OpenPCDet config snapshot into the output directory."""
    source = work_dir / Path(relative_to_openpcdet(cfg_file)).name
    if not source.exists():
        source = cfg_file
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tensorboard(work_dir: Path, target: Path) -> None:
    """Copy TensorBoard event files from the OpenPCDet work directory."""
    source = work_dir / "tensorboard"
    if not source.exists():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def read_train_eval_metrics(work_dir: Path, epoch: int) -> JsonDict:
    """Read validation metrics for one OpenPCDet epoch from disk."""
    result_root = work_dir / "eval" / "eval_with_train" / f"epoch_{epoch}"
    candidates = sorted(result_root.glob("*/metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No training-eval metrics found for epoch {epoch} under {result_root}")
    return read_json(candidates[0])


def select_best_validation_checkpoint(
    work_dir: Path,
    checkpoints: Sequence[Path],
    best_metric: str,
) -> JsonDict:
    """Choose the best checkpoint according to the selected validation metric."""
    best_item: Optional[JsonDict] = None
    best_value = float("-inf")

    for checkpoint in checkpoints:
        epoch = checkpoint_epoch(checkpoint)
        metrics_payload = read_train_eval_metrics(work_dir, epoch)
        metrics = metrics_payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        metric_value = float(metrics.get(best_metric, float("-inf")))
        item: JsonDict = {
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
    best_item: JsonDict,
    target_dir: Path,
    keep_all: bool,
) -> Optional[Path]:
    """Copy the selected best and last checkpoints into the output directory."""
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
    run_name: str,
    class_filter: str,
    dataset_name: str,
    preset: Optional[str],
    cfg_file: Path,
    set_cfgs: Optional[Sequence[str]],
    epochs: Optional[int],
    batch_size: Optional[int],
    workers: int,
    keep_all_ckpt: bool,
    pretrained_model: Optional[Path],
    resume_checkpoint: Optional[Path],
    best_metric: str,
    best_item: JsonDict,
    epochs_dir: Optional[Path],
    epoch_checkpoints: Sequence[Path],
) -> JsonDict:
    """Build the metadata payload for an OpenPCDet training run."""
    return {
        "mode": "train",
        "run": run_name,
        "dataset": {
            "family": class_filter,
            "name": dataset_name,
        },
        "config": {
            "preset": preset,
            "source": repo_relative_or_absolute(cfg_file),
            "overrides": list(set_cfgs or []),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "checkpoint_interval": 1,
            "keep_all_ckpt": keep_all_ckpt,
            "pretrained_model": repo_relative_or_absolute(pretrained_model) if pretrained_model is not None else None,
            "resume_checkpoint": (
                repo_relative_or_absolute(resume_checkpoint) if resume_checkpoint is not None else None
            ),
        },
        "validation": {
            "split": "val",
            "best_metric": best_metric,
            "best_epoch": best_item["epoch"],
            "best_value": best_item["metric_value"],
        },
        "checkpoints": {
            "best_epoch": best_item["epoch"],
            "last_epoch": checkpoint_epoch(epoch_checkpoints[-1]),
            "kept_epoch_count": len(epoch_checkpoints) if epochs_dir is not None else 0,
        },
    }


def build_test_meta(
    *,
    run_name: str,
    class_filter: str,
    dataset_name: str,
    preset: Optional[str],
    set_cfgs: Optional[Sequence[str]],
    batch_size: Optional[int],
    workers: int,
    cfg_file: Path,
    checkpoint: Path,
) -> JsonDict:
    """Build the metadata payload for an OpenPCDet evaluation run."""
    epoch = checkpoint_epoch(checkpoint)
    return {
        "mode": "test",
        "run": run_name,
        "dataset": {
            "family": class_filter,
            "name": dataset_name,
            "split": "test",
        },
        "config": {
            "preset": preset,
            "source": repo_relative_or_absolute(cfg_file),
            "overrides": list(set_cfgs or []),
        },
        "checkpoint": {
            "path": repo_relative_or_absolute(checkpoint),
            "epoch": epoch if epoch >= 0 else None,
        },
        "evaluation": {
            "batch_size": batch_size,
            "workers": workers,
        },
    }


def make_checkpoint_link(checkpoint: Path, epoch: str, tmp_dir: Path) -> Path:
    """Create a temporary symlink to a checkpoint using the expected upstream naming scheme."""
    name = "checkpoint_no_number.pth" if epoch == "no_number" else f"checkpoint_epoch_{epoch}.pth"
    link_path = tmp_dir / name
    os.symlink(checkpoint, link_path)
    return link_path


def copy_test_artifacts(work_dir: Path, output_dir: Path, checkpoint: Path) -> None:
    """Copy OpenPCDet evaluation artifacts into the final output directory."""
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
