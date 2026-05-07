import json
from pathlib import Path


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
            stem = stem[len(prefix) :]
            break
    try:
        return int(float(stem))
    except ValueError:
        return -1


def latest_file(paths):
    paths = list(paths)
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None
