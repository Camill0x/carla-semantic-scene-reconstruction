from typing import Callable, Optional

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from gui.models import FlagSpec
from gui.types import ArgsList
from gui.widgets.flag_form import FlagForm


class ProcessPanel(QFrame):
    def __init__(
        self,
        *,
        title: str,
        description: str,
        flags: list[FlagSpec],
        on_start: Callable[[ArgsList], None],
        on_stop: Callable[[], None],
        on_restart: Optional[Callable[[ArgsList], None]] = None,
        allow_extra_args: bool,
        initial_args: Optional[ArgsList] = None,
    ) -> None:
        """Build the process control panel for one managed command."""
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_restart = on_restart
        self.form = FlagForm(flags, allow_extra_args)

        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #8ea0bd;")
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-weight: 700; color: #9aa5b1;")
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.form)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.restart_button = QPushButton("Restart")
        self.stop_button = QPushButton("Stop")
        self.start_button.setProperty("buttonRole", "success")
        self.restart_button.setProperty("buttonRole", "warning")
        self.stop_button.setProperty("buttonRole", "danger")
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self.on_stop)
        self.restart_button.clicked.connect(self._restart)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.restart_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.restart_button.setVisible(on_restart is not None)
        if initial_args:
            self.form.set_from_args(initial_args)

    def _start(self) -> None:
        """Start the managed process using the current form values."""
        self.on_start(self.form.to_args())

    def _restart(self) -> None:
        """Restart the managed process using the current form values."""
        if self.on_restart is not None:
            self.on_restart(self.form.to_args())

    def set_status(self, status: str) -> None:
        """Update the process status label and its color styling."""
        self.status_label.setText(f"Status: {status}")
        if status == "Running":
            color = "#5fd08d"
        elif status.startswith("Closed"):
            color = "#f2a65a"
        else:
            color = "#9aa5b1"
        self.status_label.setStyleSheet(f"font-weight: 700; color: {color};")

    def load_values(self, values: dict[str, object]) -> None:
        """Populate the process form from a saved value dictionary."""
        self.form.set_values(values)

    def validation_error(self) -> Optional[str]:
        """Return the first validation error for the current process form, if any."""
        return self.form.validation_error()
