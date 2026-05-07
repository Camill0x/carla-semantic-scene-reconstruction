import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

from src.common.paths import repo_path
from src.openpcdet.constants import CLASS_FILTERS, OPENPCDET_PRESETS

OPENPCDET_ROOT = repo_path("third_party", "OpenPCDet")
OPENPCDET_CFG_ROOT = OPENPCDET_ROOT / "tools" / "cfgs"
RAW_DATASET_ROOT = repo_path("datasets", "raw")
DATASET_ROOT = repo_path("datasets", "openpcdet")
RESULTS_ROOT = repo_path("results", "openpcdet")


def cfg_file_class_filter(cfg_file: Path) -> str:
    cfg_dir = cfg_file.parent.name
    for class_filter in CLASS_FILTERS:
        if cfg_dir == f"{class_filter}_models":
            return class_filter
    return next(iter(CLASS_FILTERS))


def resolve_openpcdet_preset(preset: str) -> Tuple[str, Path]:
    if preset not in OPENPCDET_PRESETS:
        raise ValueError(f"Unknown OpenPCDet preset: {preset}")
    class_filter, cfg_file = OPENPCDET_PRESETS[preset]
    cfg_path = OPENPCDET_CFG_ROOT / f"{class_filter}_models" / cfg_file
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    return class_filter, cfg_path


def relative_to_openpcdet(path: Path) -> str:
    return os.path.relpath(Path(path), OPENPCDET_ROOT)


def prepared_dataset_root(class_filter: str, dataset_name: str) -> Path:
    return DATASET_ROOT / class_filter / dataset_name


def prepared_dataset_data_path(class_filter: str, dataset_name: str) -> str:
    return relative_to_openpcdet(prepared_dataset_root(class_filter, dataset_name))


def generated_run_name(parent_dir: Path) -> str:
    base_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not (parent_dir / base_name).exists():
        return base_name

    index = 2
    while True:
        candidate = f"{base_name}_{index}"
        if not (parent_dir / candidate).exists():
            return candidate
        index += 1
