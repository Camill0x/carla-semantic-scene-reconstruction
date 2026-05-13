from typing import Callable, TypeAlias, Union

ArgsList: TypeAlias = list[str]
ActivityMessages: TypeAlias = Union[str, list[str]]
AppendActivity: TypeAlias = Callable[[ActivityMessages], None]
OpenWorkflow: TypeAlias = Callable[[str], None]
SummarySpec: TypeAlias = tuple[str, str]
SummaryValues: TypeAlias = dict[str, str]
StatusRow: TypeAlias = dict[str, str]
StatusRows: TypeAlias = list[StatusRow]
