import json
from pathlib import Path
from typing import Iterable, Optional, TypeVar, cast

from src.common.typing_aliases import JsonDict

PathT = TypeVar("PathT", bound=Path)


def read_json(path: Path) -> JsonDict:
    """Read a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(JsonDict, payload)


def write_json(path: Path, payload: JsonDict) -> None:
    """Write a JSON payload to disk with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def checkpoint_epoch(path: Path) -> int:
    """Extract the epoch number encoded in a checkpoint filename."""
    stem = Path(path).stem
    for prefix in ("checkpoint_epoch_", "epoch_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    try:
        return int(float(stem))
    except ValueError:
        return -1


def latest_file(paths: Iterable[PathT]) -> Optional[PathT]:
    """Return the newest file from an iterable of candidate paths."""
    paths = list(paths)
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None
