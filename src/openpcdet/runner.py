import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from src.openpcdet.paths import OPENPCDET_ROOT


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def run_openpcdet_tool(tool_name: str, args: Sequence[str]) -> int:
    command = [sys.executable, str(OPENPCDET_ROOT / "tools" / tool_name), *args]
    completed = subprocess.run(command, cwd=OPENPCDET_ROOT, check=False)
    return int(completed.returncode)


def extend_with_set_args(base_args: List[str], set_cfgs: Optional[Iterable[str]]) -> List[str]:
    if set_cfgs:
        base_args.extend(["--set", *set_cfgs])
    return base_args
