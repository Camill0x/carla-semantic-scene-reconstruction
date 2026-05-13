import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, List, Optional, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DatasetSplits(Generic[T]):
    train: List[T]
    val: List[T]
    test: List[T]


def selected_run_dirs(source_root: Path, run_names: Optional[Sequence[str]]) -> List[Path]:
    """Return the selected raw dataset run directories in sorted order."""
    if run_names is None:
        return sorted(path for path in source_root.iterdir() if path.is_dir() and path.name.startswith("run_"))

    run_dirs = []
    for run_name in run_names:
        run_dir = source_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        if not run_dir.is_dir():
            raise NotADirectoryError(run_dir)
        run_dirs.append(run_dir)
    return run_dirs


def iter_frame_dirs(run_dirs: Sequence[Path]) -> List[Path]:
    """Return the sorted frame directories for the provided runs."""
    return [
        frame_dir
        for run_dir in run_dirs
        for frame_dir in sorted(run_dir.iterdir())
        if frame_dir.is_dir() and frame_dir.name.startswith("frame_")
    ]


def train_val_test_split(items: Sequence[T], val_ratio: float, test_ratio: float, seed: int) -> DatasetSplits[T]:
    """Split a sequence into shuffled train, validation, and test subsets."""
    if val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError("Split ratios must be >= 0")
    if val_ratio + test_ratio >= 1.0 and len(items) > 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)

    num_test = int(round(len(indices) * test_ratio))
    num_val = int(round(len(indices) * val_ratio))

    if len(indices) >= 3:
        num_test = max(1, num_test) if test_ratio > 0.0 else 0
        num_val = max(1, num_val) if val_ratio > 0.0 else 0
        while num_test + num_val >= len(indices):
            if num_val >= num_test and num_val > 0:
                num_val -= 1
            elif num_test > 0:
                num_test -= 1
            else:
                break
    else:
        num_test = min(num_test, max(0, len(indices) - 1))
        num_val = min(num_val, max(0, len(indices) - num_test - 1))

    test_indices = set(indices[:num_test])
    val_indices = set(indices[num_test : num_test + num_val])

    train = []
    val = []
    test = []

    for index, item in enumerate(items):
        if index in test_indices:
            test.append(item)
        elif index in val_indices:
            val.append(item)
        else:
            train.append(item)

    return DatasetSplits(train=train, val=val, test=test)
