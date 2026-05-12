from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class FlagSpec:
    key: str
    label: str
    arg: str
    kind: str = "text"
    default: Any = None
    help_text: str = ""
    required: bool = False
    placeholder: str = ""
    choices: Optional[List[str]] = None
    enabled: bool = True


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    title: str
    command: List[str]
