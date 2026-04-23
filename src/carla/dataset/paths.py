from pathlib import Path


def ensure_dataset_run_dir(root_dir: Path) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)

    existing_indices = []
    for child in root_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("run_"):
            continue
        suffix = child.name[len("run_") :]
        if suffix.isdigit():
            existing_indices.append(int(suffix))

    next_index = max(existing_indices, default=0) + 1
    run_dir = root_dir / f"run_{next_index:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
