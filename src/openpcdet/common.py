import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.common.paths import repo_relative_or_absolute
from src.openpcdet.artifacts import checkpoint_epoch, latest_file, read_json, write_json
from src.openpcdet.paths import (
    config_dir,
    evaluations_dir,
    relative_to_openpcdet,
    run_dir,
    summary_path,
)


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def append_log_summary(log_path: Path, lines: Sequence[str]) -> None:
    if not log_path.exists():
        return
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n\nFinal project artifacts:\n")
        for line in lines:
            handle.write(f"- {line}\n")


def print_artifact_summary(lines: Sequence[str]) -> None:
    print("\nFinal project artifacts:")
    for line in lines:
        print(f"- {line}")


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


def collect_validation_history(
    work_dir: Path,
    checkpoints: Sequence[Path],
    best_metric: str,
) -> Tuple[List[Dict], Dict]:
    val_history = []
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
        val_history.append(item)
        if best_item is None or metric_value > best_value:
            best_value = metric_value
            best_item = item

    if best_item is None:
        raise RuntimeError("Validation did not produce a best checkpoint")
    return val_history, best_item


def copy_selected_checkpoints(
    checkpoints: Sequence[Path],
    best_item: Dict,
    target_dir: Path,
    keep_all: bool,
) -> Tuple[Path, Path, Optional[Path], Path]:
    best_source = next(path for path in checkpoints if checkpoint_epoch(path) == best_item["epoch"])
    last_source = checkpoints[-1]
    best_target = target_dir / "best.ckpt"
    last_target = target_dir / "last.ckpt"
    shutil.copy2(best_source, best_target)
    shutil.copy2(last_source, last_target)

    epochs_dir = None
    if keep_all:
        epochs_dir = target_dir / "epochs"
        recreate_dir(epochs_dir)
        for checkpoint in checkpoints:
            shutil.copy2(checkpoint, epochs_dir / f"epoch_{checkpoint_epoch(checkpoint):03d}.ckpt")

    return best_target, last_target, epochs_dir, last_source


def build_training_summary(
    *,
    args: Namespace,
    cfg_file: Path,
    best_item: Dict,
    last_source: Path,
    best_target: Path,
    last_target: Path,
    epochs_dir: Optional[Path],
    epoch_checkpoints: Sequence[Path],
    output_dir: Path,
    eval_dir: Path,
    log_dir: Path,
) -> Dict:
    return {
        "name": args.name,
        "config": {
            "source_cfg": repo_relative_or_absolute(cfg_file),
            "used_cfg": repo_relative_or_absolute(config_dir(args.name) / "used.yaml"),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "best_metric": args.best_metric,
            "checkpoint_interval": 1,
            "keep_all_ckpt": args.keep_all_ckpt,
            "pretrained_model": (
                repo_relative_or_absolute(args.pretrained_model) if args.pretrained_model is not None else None
            ),
            "resume_checkpoint": repo_relative_or_absolute(args.ckpt) if args.ckpt is not None else None,
            "dataset_name": args.dataset_name,
        },
        "artifacts": {
            "checkpoints": {
                "best": repo_relative_or_absolute(best_target),
                "last": repo_relative_or_absolute(last_target),
                "epochs_dir": repo_relative_or_absolute(epochs_dir) if epochs_dir is not None else None,
                "kept_epoch_count": len(epoch_checkpoints) if epochs_dir is not None else 0,
            },
            "logs": {
                "train": repo_relative_or_absolute(log_dir / "train.log"),
            },
            "evaluations": {
                "val_history": repo_relative_or_absolute(eval_dir / "val_history.json"),
                "best_val": repo_relative_or_absolute(eval_dir / "best_val.json"),
            },
            "tensorboard_dir": repo_relative_or_absolute(output_dir / "tensorboard"),
        },
        "selection": {
            "best": {
                "epoch": best_item["epoch"],
                "metric": args.best_metric,
                "value": best_item["metric_value"],
            },
            "last": {
                "epoch": checkpoint_epoch(last_source),
            },
        },
    }


def training_final_lines(
    best_target: Path,
    last_target: Path,
    eval_dir: Path,
    summary_json_path: Path,
    tensorboard_dir: Path,
) -> List[str]:
    return [
        f"best_checkpoint: {repo_relative_or_absolute(best_target)}",
        f"last_checkpoint: {repo_relative_or_absolute(last_target)}",
        f"val_history: {repo_relative_or_absolute(eval_dir / 'val_history.json')}",
        f"best_val: {repo_relative_or_absolute(eval_dir / 'best_val.json')}",
        f"summary: {repo_relative_or_absolute(summary_json_path)}",
        f"tensorboard: {repo_relative_or_absolute(tensorboard_dir)}",
    ]


def selected_epoch(args: Namespace, checkpoint: Path) -> str:
    if args.checkpoint == "best" or checkpoint.name == "best.ckpt":
        best_val_path = evaluations_dir(args.name) / "best_val.json"
        if best_val_path.exists():
            return str(read_json(best_val_path)["epoch"])

    if args.checkpoint == "last" or checkpoint.name == "last.ckpt":
        run_summary_path = summary_path(args.name)
        if run_summary_path.exists():
            return str(read_json(run_summary_path)["selection"]["last"]["epoch"])

    epoch = checkpoint_epoch(checkpoint)
    return str(epoch) if epoch >= 0 else "no_number"


def make_checkpoint_link(checkpoint: Path, epoch: str, tmp_dir: Path) -> Path:
    name = "checkpoint_no_number.pth" if epoch == "no_number" else f"checkpoint_epoch_{epoch}.pth"
    link_path = tmp_dir / name
    os.symlink(checkpoint, link_path)
    return link_path


def default_eval_tag(selector: str, epoch: str) -> str:
    if selector in {"best", "last"}:
        return selector
    if epoch != "no_number":
        return f"epoch_{int(epoch):03d}"
    return Path(selector).stem


def copy_eval_artifacts(work_dir: Path, run_name: str, eval_tag: str, checkpoint: Path) -> None:
    target_dir = evaluations_dir(run_name) / eval_tag
    recreate_dir(target_dir)

    metrics_path = latest_file(work_dir.glob(f"eval/**/{eval_tag}/metrics.json"))
    result_path = latest_file(work_dir.glob(f"eval/**/{eval_tag}/result.pkl"))
    log_path = latest_file(work_dir.glob(f"eval/**/{eval_tag}/log_eval_*.txt"))

    if metrics_path is None:
        raise FileNotFoundError(f"No metrics.json produced under {work_dir}")

    metrics = read_json(metrics_path)
    metrics["checkpoint"] = repo_relative_or_absolute(checkpoint)
    write_json(target_dir / "metrics.json", metrics)
    if result_path is not None:
        shutil.copy2(result_path, target_dir / "result.pkl")
    if log_path is not None:
        logs_dir = run_dir(run_name) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_name = "test.log" if eval_tag == "test" else f"test_{eval_tag}.log"
        final_log_path = logs_dir / log_name
        shutil.copy2(log_path, final_log_path)
        final_lines = [
            f"checkpoint: {repo_relative_or_absolute(checkpoint)}",
            f"metrics: {repo_relative_or_absolute(target_dir / 'metrics.json')}",
            (
                f"result_pkl: {repo_relative_or_absolute(target_dir / 'result.pkl')}"
                if result_path is not None
                else "result_pkl: not produced"
            ),
        ]
        append_log_summary(final_log_path, final_lines)
        print_artifact_summary(final_lines)
