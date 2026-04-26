from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    return repo_root().joinpath(*parts)


def repo_relative_or_absolute(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root()))
    except ValueError:
        return str(resolved)


def validate_directory_name(name: str) -> str:
    if "/" in name or "\\" in name:
        raise ValueError("Name must be a single directory name, not a path")
    return name
