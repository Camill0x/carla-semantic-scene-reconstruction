import shlex
from pathlib import Path
from typing import Dict, List, Optional, TypeAlias

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.models import FlagSpec

InputWidget: TypeAlias = QCheckBox | QComboBox | QLineEdit


class FlagForm(QWidget):
    def __init__(self, flags: List[FlagSpec], allow_extra_args: bool) -> None:
        super().__init__()
        self.flags = list(flags)
        self.allow_extra_args = allow_extra_args
        self.inputs: Dict[str, InputWidget] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(10)
        layout.addLayout(form)

        for flag in self.flags:
            widget = self._build_input(flag)
            form.addRow(self._build_label(flag), widget)

        self.extra_args: Optional[QLineEdit] = None
        if self.allow_extra_args:
            extras_label = QLabel("Additional Args")
            extras_label.setToolTip("Extra raw CLI arguments appended at the end.")
            self.extra_args = QLineEdit()
            self.extra_args.setPlaceholderText("Optional extra CLI args")
            form.addRow(extras_label, self.extra_args)

    def _build_label(self, flag: FlagSpec) -> QLabel:
        label = QLabel(flag.label)
        if flag.help_text:
            label.setToolTip(flag.help_text)
        return label

    def _build_input(self, flag: FlagSpec) -> QWidget:
        if flag.kind == "bool":
            checkbox = QCheckBox()
            checkbox.setChecked(bool(flag.default))
            checkbox.setEnabled(flag.enabled)
            self.inputs[flag.key] = checkbox
            return checkbox
        if flag.kind == "choice":
            widget = QComboBox()
            for choice in flag.choices or []:
                widget.addItem(choice)
            if flag.default in (flag.choices or []):
                widget.setCurrentText(str(flag.default))
            widget.setEnabled(flag.enabled)
            self.inputs[flag.key] = widget
            return widget
        if flag.kind in {"file", "dir"}:
            return self._build_path_row(flag)

        line_edit = QLineEdit()
        if flag.default not in (None, ""):
            line_edit.setText(str(flag.default))
        if flag.placeholder:
            line_edit.setPlaceholderText(flag.placeholder)
        line_edit.setEnabled(flag.enabled)
        self.inputs[flag.key] = line_edit
        return line_edit

    def _build_path_row(self, flag: FlagSpec) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        line_edit = QLineEdit()
        if flag.default not in (None, ""):
            line_edit.setText(str(flag.default))
        line_edit.setEnabled(flag.enabled)
        browse = QPushButton("Browse")
        browse.setProperty("buttonRole", "secondary")
        browse.clicked.connect(lambda: self._browse(flag.kind, line_edit))
        browse.setEnabled(flag.enabled)
        row.addWidget(line_edit, 1)
        row.addWidget(browse)
        self.inputs[flag.key] = line_edit
        return container

    def _browse(self, kind: str, line_edit: QLineEdit) -> None:
        current = line_edit.text().strip()
        start_dir = str(Path(current).expanduser().resolve().parent) if current else ""
        if kind == "dir":
            chosen = QFileDialog.getExistingDirectory(self, "Choose directory", start_dir)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "Choose file", start_dir)
        if chosen:
            line_edit.setText(chosen)

    def values(self) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        for flag in self.flags:
            widget = self.inputs[flag.key]
            if isinstance(widget, QCheckBox):
                payload[flag.key] = bool(widget.isChecked())
            elif isinstance(widget, QComboBox):
                payload[flag.key] = str(widget.currentText()).strip()
            else:
                payload[flag.key] = str(widget.text()).strip()
        payload["__extra_args__"] = self.extra_args.text().strip() if self.extra_args is not None else ""
        return payload

    def set_values(self, values: Dict[str, object]) -> None:
        for flag in self.flags:
            if flag.key not in values:
                continue
            widget = self.inputs[flag.key]
            value = values[flag.key]
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                text = str(value)
                if text:
                    widget.setCurrentText(text)
            else:
                widget.setText("" if value is None else str(value))
        extra = values.get("__extra_args__")
        if extra is not None and self.extra_args is not None:
            self.extra_args.setText(str(extra))

    def set_from_args(self, args: List[str]) -> None:
        parsed: Dict[str, object] = {}
        extra_tokens: List[str] = []
        flag_by_arg = {flag.arg: flag for flag in self.flags}
        index = 0

        while index < len(args):
            token = str(args[index])
            flag = flag_by_arg.get(token)
            if flag is None:
                extra_tokens.append(token)
                index += 1
                continue

            if flag.kind == "bool":
                parsed[flag.key] = True
                index += 1
                continue

            if flag.key in {"runs", "gpus"}:
                values: List[str] = []
                cursor = index + 1
                while cursor < len(args) and str(args[cursor]) not in flag_by_arg:
                    values.append(str(args[cursor]))
                    cursor += 1
                parsed[flag.key] = " ".join(values)
                index = cursor
                continue

            if index + 1 < len(args):
                parsed[flag.key] = str(args[index + 1])
                index += 2
                continue

            extra_tokens.append(token)
            index += 1

        if extra_tokens:
            parsed["__extra_args__"] = shlex.join(extra_tokens)

        self.set_values(parsed)

    def to_args(self) -> List[str]:
        args: List[str] = []
        values = self.values()
        config_file_selected = any(str(values.get(key, "")).strip() for key in ["cfg_file", "config"])
        runs_selected = bool(str(values.get("runs", "")).strip())
        for flag in self.flags:
            value = values.get(flag.key)
            if flag.kind == "bool":
                if flag.key == "all_runs" and runs_selected:
                    continue
                if value:
                    args.append(flag.arg)
                continue

            text = "" if value is None else str(value).strip()
            if not text:
                continue
            if flag.key == "preset" and config_file_selected:
                continue

            if flag.key == "runs":
                args.append(flag.arg)
                args.extend(shlex.split(text))
                continue
            if flag.key == "gpus":
                args.append(flag.arg)
                args.extend(shlex.split(text))
                continue

            args.extend([flag.arg, text])

        extra = str(values.get("__extra_args__", "")).strip()
        if extra:
            args.extend(shlex.split(extra))
        return args

    def validation_error(self) -> Optional[str]:
        values = self.values()
        for flag in self.flags:
            value = values.get(flag.key)
            text = "" if value is None else str(value).strip()
            if flag.required and not text:
                return f"{flag.label} is required."
        return None
