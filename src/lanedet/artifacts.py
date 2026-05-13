import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from src.common.paths import repo_relative_or_absolute
from src.common.typing_aliases import JsonDict
from src.lanedet.metrics import build_tusimple_metrics


def latest_lanedet_work_dir(work_root: Path) -> Path:
    candidates = sorted((work_root / "TuSimple").glob("*"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No LaneDet work directory produced under {work_root / 'TuSimple'}")
    return candidates[-1]


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_log(source: Path, target: Path) -> Optional[Path]:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def copy_config(source_work_dir: Path, target: Path) -> Path:
    source = source_work_dir / "config.py"
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def copy_train_checkpoints(source_work_dir: Path, target_dir: Path) -> Dict[str, Optional[Path]]:
    source_ckpt_dir = source_work_dir / "ckpt"
    if not source_ckpt_dir.exists():
        raise FileNotFoundError(source_ckpt_dir)

    recreate_dir(target_dir)
    best_target = None
    last_target = None

    best_source = source_ckpt_dir / "best.pth"
    if best_source.exists():
        best_target = target_dir / "best.pth"
        shutil.copy2(best_source, best_target)

    numeric_sources = sorted(
        [path for path in source_ckpt_dir.glob("*.pth") if path.stem.isdigit()],
        key=lambda path: int(path.stem),
    )
    if numeric_sources:
        last_target = target_dir / "last.pth"
        shutil.copy2(numeric_sources[-1], last_target)

    return {"best": best_target, "last": last_target}


def copy_common_outputs(source_work_dir: Path, output_dir: Path, gt_json: Path) -> None:
    copy_config(source_work_dir, output_dir / "config.py")

    predictions_source = source_work_dir / "tusimple_predictions.json"
    if predictions_source.exists():
        predictions_target = output_dir / "predictions.json"
        shutil.copy2(predictions_source, predictions_target)
        write_json(output_dir / "metrics.json", build_tusimple_metrics(predictions_target, gt_json))

    visualization_source = source_work_dir / "visualization"
    if visualization_source.exists():
        shutil.copytree(visualization_source, output_dir / "visualization")


def write_run_metadata(
    *,
    output_dir: Path,
    mode: str,
    run_name: str,
    dataset: str,
    model: str,
    data_root: Optional[Path],
    preset: Optional[str],
    source_config: Path,
    load_from: Optional[Path],
    finetune_from: Optional[Path],
) -> Path:
    payload: JsonDict = {
        "mode": mode,
        "run": run_name,
        "dataset": dataset,
        "model": model,
        "data_root": repo_relative_or_absolute(data_root) if data_root is not None else None,
        "preset": preset,
        "source_config": repo_relative_or_absolute(source_config),
        "load_from": repo_relative_or_absolute(load_from) if load_from is not None else None,
        "finetune_from": repo_relative_or_absolute(finetune_from) if finetune_from is not None else None,
    }
    return write_json(output_dir / "meta.json", payload)


def write_json(path: Path, payload: JsonDict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
