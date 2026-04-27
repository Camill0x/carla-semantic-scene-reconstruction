from pathlib import Path

from src.common.paths import repo_path, validate_directory_name
from src.lanedet.constants import (
    DEFAULT_DATASET_NAME,
    LANEDET_DATASET_FAMILY,
    LANEDET_PRESETS,
    LANEDET_RESULTS_DIR,
    LANEDET_THIRD_PARTY_DIR,
)


def lanedet_root() -> Path:
    return repo_path(*LANEDET_THIRD_PARTY_DIR)


def lanedet_config_dir() -> Path:
    return lanedet_root() / "configs"


def resolve_lanedet_cfg(preset: str) -> Path:
    try:
        return lanedet_root().joinpath(*LANEDET_PRESETS[preset])
    except KeyError as exc:
        raise ValueError(f"Unknown LaneDet preset: {preset}") from exc


def raw_dataset_root() -> Path:
    return repo_path("datasets", "raw")


def dataset_root(dataset: str = LANEDET_DATASET_FAMILY) -> Path:
    return repo_path("datasets", LANEDET_RESULTS_DIR, validate_directory_name(dataset))


def prepared_dataset_root(name: str = DEFAULT_DATASET_NAME, dataset: str = LANEDET_DATASET_FAMILY) -> Path:
    return dataset_root(dataset) / validate_directory_name(name)


def results_root(dataset: str = LANEDET_DATASET_FAMILY) -> Path:
    return repo_path("results", LANEDET_RESULTS_DIR, validate_directory_name(dataset))


def run_dir(run_name: str, dataset: str = LANEDET_DATASET_FAMILY) -> Path:
    return results_root(dataset) / validate_directory_name(run_name)


def checkpoints_dir(run_name: str, dataset: str = LANEDET_DATASET_FAMILY) -> Path:
    return run_dir(run_name, dataset) / "ckpt"


def evaluations_dir(run_name: str, dataset: str = LANEDET_DATASET_FAMILY) -> Path:
    return run_dir(run_name, dataset) / "evaluations"


def best_checkpoint_path(run_name: str, dataset: str = LANEDET_DATASET_FAMILY) -> Path:
    return checkpoints_dir(run_name, dataset) / "best.pth"


def last_checkpoint_path(run_name: str, dataset: str = LANEDET_DATASET_FAMILY) -> Path:
    return checkpoints_dir(run_name, dataset) / "last.pth"
