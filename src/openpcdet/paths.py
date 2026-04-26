import os
from pathlib import Path

from src.common.paths import repo_path, validate_directory_name
from src.openpcdet.constants import (
    DATASET_FAMILY,
    DEFAULT_DATASET_NAME,
    OPENPCDET_PRESETS,
    OPENPCDET_RESULTS_DIR,
    OPENPCDET_THIRD_PARTY_DIR,
)


def openpcdet_root() -> Path:
    return repo_path(*OPENPCDET_THIRD_PARTY_DIR)


def openpcdet_models_cfg_dir() -> Path:
    return openpcdet_root() / "tools" / "cfgs" / f"{DATASET_FAMILY}_models"


def resolve_openpcdet_cfg(preset: str) -> Path:
    try:
        return openpcdet_models_cfg_dir() / OPENPCDET_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"Unknown OpenPCDet preset: {preset}") from exc


def relative_to_openpcdet(path: Path) -> str:
    return os.path.relpath(Path(path), openpcdet_root())


def raw_dataset_root() -> Path:
    return repo_path("datasets", "raw")


def dataset_root() -> Path:
    return repo_path("datasets", OPENPCDET_RESULTS_DIR, DATASET_FAMILY)


def prepared_dataset_root(name: str = DEFAULT_DATASET_NAME) -> Path:
    return dataset_root() / validate_directory_name(name)


def prepared_dataset_data_path(name: str = DEFAULT_DATASET_NAME) -> str:
    return relative_to_openpcdet(prepared_dataset_root(name))


def results_root() -> Path:
    return repo_path("results", OPENPCDET_RESULTS_DIR)


def run_dir(run_name: str) -> Path:
    return results_root() / validate_directory_name(run_name)


def config_dir(run_name: str) -> Path:
    return run_dir(run_name) / "config"


def checkpoints_dir(run_name: str) -> Path:
    return run_dir(run_name) / "checkpoints"


def evaluations_dir(run_name: str) -> Path:
    return run_dir(run_name) / "evaluations"


def best_checkpoint_path(run_name: str) -> Path:
    return checkpoints_dir(run_name) / "best.ckpt"


def last_checkpoint_path(run_name: str) -> Path:
    return checkpoints_dir(run_name) / "last.ckpt"


def summary_path(run_name: str) -> Path:
    return run_dir(run_name) / "summary.json"
