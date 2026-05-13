from pathlib import Path


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    """Return a repository-relative path anchored at the project root."""
    return repo_root().joinpath(*parts)


def repo_relative_or_absolute(path: Path) -> str:
    """Return a repository-relative path when possible, otherwise an absolute path."""
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root()))
    except ValueError:
        return str(resolved)
