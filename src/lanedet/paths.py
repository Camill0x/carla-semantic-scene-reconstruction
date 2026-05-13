import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

from src.common.paths import repo_path, repo_relative_or_absolute, repo_root
from src.lanedet.constants import LANEDET_PRESETS

LANEDET_ROOT = repo_path("third_party", "lanedet")
RAW_DATASET_ROOT = repo_path("datasets", "raw")
DATASET_ROOT = repo_path("datasets", "lanedet")
RESULTS_ROOT = repo_path("results", "lanedet")


def resolve_lanedet_cfg(preset: str) -> Path:
    """Resolve a named LaneDet preset to its config file."""
    _, model, cfg_name = resolve_lanedet_preset(preset)
    return LANEDET_ROOT / "configs" / model / cfg_name


def preset_dataset(preset: str) -> str:
    """Return the dataset family associated with a LaneDet preset."""
    dataset, _, _ = resolve_lanedet_preset(preset)
    return dataset


def preset_model(preset: str) -> str:
    """Return the model family associated with a LaneDet preset."""
    _, model, _ = resolve_lanedet_preset(preset)
    return model


def resolve_lanedet_preset(preset: str) -> Tuple[str, str, str]:
    """Resolve a LaneDet preset into its config, model, and dataset identifiers."""
    try:
        return LANEDET_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"Unknown LaneDet preset: {preset}") from exc


def model_from_config_path(config_path: Path) -> str:
    """Infer the LaneDet model name from a config path."""
    resolved = config_path.expanduser().resolve()
    cfg_root = LANEDET_ROOT / "configs"
    try:
        relative_cfg = resolved.relative_to(cfg_root)
    except ValueError:
        relative_cfg = None
    if relative_cfg is not None and len(relative_cfg.parts) >= 2:
        return relative_cfg.parts[0]

    meta_path = resolved.with_name("meta.json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("model"):
            return str(meta["model"])
        if meta.get("preset"):
            return preset_model(str(meta["preset"]))
        if meta.get("source_config"):
            source_config = Path(str(meta["source_config"]))
            if not source_config.is_absolute():
                source_config = repo_root() / source_config
            return model_from_config_path(source_config)

    raise ValueError(f"Could not infer LaneDet model from config path: {repo_relative_or_absolute(resolved)}")


def generated_run_name() -> str:
    """Generate a timestamp-based run name that does not collide in the target directory."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepared_dataset_root(name: str = "default", dataset: str = "tusimple") -> Path:
    """Return the root directory of the prepared dataset variant."""
    return DATASET_ROOT / dataset / name


def results_root(mode: str, model: str, dataset: str = "tusimple") -> Path:
    """Return the LaneDet results root for the selected mode and model."""
    return RESULTS_ROOT / mode / model / dataset


def run_dir(run_name: str, mode: str, model: str, dataset: str = "tusimple") -> Path:
    """Return the LaneDet run directory for the selected mode and model."""
    return results_root(mode, model, dataset) / run_name


def checkpoints_dir(run_name: str, mode: str, model: str, dataset: str = "tusimple") -> Path:
    """Return the checkpoint directory for a LaneDet run."""
    return run_dir(run_name, mode, model, dataset) / "ckpt"
