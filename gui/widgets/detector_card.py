from typing import Callable

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from pathlib import Path


class DetectorCard(QGroupBox):
    def __init__(
        self,
        *,
        detector_name: str,
        title: str,
        config_label: str,
        on_start: Callable[[str, str, str], None],
        on_stop: Callable[[str], None],
        on_change: Callable[[str, str, str], None],
    ) -> None:
        super().__init__(title)
        self.detector_name = detector_name
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_change = on_change

        self.status_label = QLabel("Stopped")
        self.status_label.setObjectName("DetectorStatus")
        self.config_input = QLineEdit()
        self.ckpt_input = QLineEdit()
        self.config_input.setPlaceholderText("Select config file")
        self.ckpt_input.setPlaceholderText("Select checkpoint file")

        layout = QVBoxLayout(self)
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        status_row.addWidget(self.status_label, 1)
        layout.addLayout(status_row)

        form = QFormLayout()
        form.addRow(config_label, self._make_browse_row(self.config_input, "Choose config/checkpoint file"))
        form.addRow("Checkpoint", self._make_browse_row(self.ckpt_input, "Choose checkpoint file"))
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.change_button = QPushButton("Change")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self._emit_start)
        self.change_button.clicked.connect(self._emit_change)
        self.stop_button.clicked.connect(self._emit_stop)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.change_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)

    def _make_browse_row(self, line_edit: QLineEdit, title: str) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse_file(line_edit, title))
        row.addWidget(line_edit, 1)
        row.addWidget(browse)
        return container

    def _browse_file(self, line_edit: QLineEdit, title: str) -> None:
        start_dir = str(Path(line_edit.text()).expanduser().resolve().parent) if line_edit.text() else ""
        file_path, _ = QFileDialog.getOpenFileName(self, title, start_dir)
        if file_path:
            line_edit.setText(file_path)

    def _emit_start(self) -> None:
        self.on_start(self.detector_name, self.config_input.text().strip(), self.ckpt_input.text().strip())

    def _emit_change(self) -> None:
        self.on_change(self.detector_name, self.config_input.text().strip(), self.ckpt_input.text().strip())

    def _emit_stop(self) -> None:
        self.on_stop(self.detector_name)

    def set_status(self, status: str) -> None:
        self.status_label.setText(status)
        color = "#1f7a1f" if status == "Running" else "#8a5a00" if status.startswith("Stopped (") else "#6b7280"
        self.status_label.setStyleSheet(f"font-weight: 700; color: {color};")

    def set_paths(self, config_path: str, ckpt_path: str) -> None:
        if config_path:
            self.config_input.setText(config_path)
        if ckpt_path:
            self.ckpt_input.setText(ckpt_path)
