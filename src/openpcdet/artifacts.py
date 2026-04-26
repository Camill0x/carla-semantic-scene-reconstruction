import json
from pathlib import Path

from src.openpcdet.paths import (
    best_checkpoint_path,
    checkpoints_dir,
    last_checkpoint_path,
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def checkpoint_epoch(path: Path) -> int:
    stem = Path(path).stem
    for prefix in ("checkpoint_epoch_", "epoch_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    try:
        return int(float(stem))
    except ValueError:
        return -1


def latest_file(paths):
    paths = list(paths)
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def resolve_checkpoint_selector(run_name: str, selector: str = "last") -> Path:
    if selector == "best":
        candidate = best_checkpoint_path(run_name)
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    if selector == "last":
        candidate = last_checkpoint_path(run_name)
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    raw_path = Path(selector).expanduser()
    if raw_path.is_absolute() or raw_path.exists():
        return raw_path.resolve()

    ckpt_dir = checkpoints_dir(run_name)
    candidates = [ckpt_dir / selector, ckpt_dir / "epochs" / selector]

    if raw_path.suffix == "":
        candidates.extend([ckpt_dir / f"{selector}.ckpt", ckpt_dir / "epochs" / f"{selector}.ckpt"])

    if selector.isdigit():
        epoch = int(selector)
        candidates.extend([
            ckpt_dir / "epochs" / f"epoch_{epoch:03d}.ckpt",
            ckpt_dir / "epochs" / f"epoch_{epoch}.ckpt",
            ckpt_dir / f"checkpoint_epoch_{epoch}.pth",
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(selector)
