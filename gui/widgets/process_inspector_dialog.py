from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

from gui.types import StatusRows
from gui.widgets.process_table import ProcessTable


class ProcessInspectorDialog(QDialog):
    def __init__(
        self,
        *,
        fetch_rows: Callable[[], StatusRows],
        on_stop_selected: Callable[[str], None],
        on_stop_all: Callable[[], None],
    ) -> None:
        super().__init__()
        self.fetch_rows = fetch_rows
        self.on_stop_selected = on_stop_selected
        self.on_stop_all = on_stop_all
        self.setWindowTitle("Process Inspector")

        layout = QVBoxLayout(self)
        self.table = ProcessTable()
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        refresh = QPushButton("Refresh")
        stop_selected = QPushButton("Stop Selected")
        stop_all = QPushButton("Stop All Running")
        close_button = QPushButton("Close")
        stop_selected.setProperty("buttonRole", "warning")
        stop_all.setProperty("buttonRole", "danger")
        close_button.setProperty("buttonRole", "danger")
        refresh.clicked.connect(self.refresh_rows)
        stop_selected.clicked.connect(self._stop_selected)
        stop_all.clicked.connect(self.on_stop_all)
        close_button.clicked.connect(self.accept)
        buttons.addWidget(refresh)
        buttons.addWidget(stop_selected)
        buttons.addWidget(stop_all)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        self.refresh_rows()
        QTimer.singleShot(0, self._resize_to_contents)

    def refresh_rows(self) -> None:
        self.table.update_rows(self.fetch_rows())
        self._resize_to_contents()

    def _stop_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        name = item.data(256) or item.text()
        self.on_stop_selected(str(name))
        self.refresh_rows()

    def _resize_to_contents(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self.table.resizeColumnsToContents()
        header = self.table.horizontalHeader()
        table_width = self.table.verticalHeader().width() + 28
        for section in range(self.table.columnCount()):
            table_width += header.sectionSize(section)
        table_width += self.table.frameWidth() * 2

        layout = self.layout()
        hint = layout.sizeHint() if layout is not None else self.sizeHint()
        width = min(max(540, max(table_width + 40, hint.width())), int(available.width() * 0.72))
        height = min(max(360, hint.height() + 24), int(available.height() * 0.75))
        self.resize(width, height)
