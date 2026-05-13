from typing import Dict, Optional

from PySide6.QtCore import QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LogViewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.log_texts: Dict[str, str] = {}
        self._known_log_names: list[str] = []
        self._last_text: Optional[str] = None
        self._force_scroll_to_bottom = True

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Log:"))
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self._handle_selection_changed)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        toolbar.addWidget(self.selector, 1)
        toolbar.addWidget(self.refresh_button)
        layout.addLayout(toolbar)

        self.editor = QPlainTextEdit()
        self.editor.setReadOnly(True)
        layout.addWidget(self.editor, 1)

    def set_logs(self, logs: Dict[str, str]) -> None:
        current = self.selector.currentText()
        self.log_texts = dict(logs)
        names = sorted(self.log_texts)
        if names != self._known_log_names:
            self._known_log_names = names
            self.selector.blockSignals(True)
            self.selector.clear()
            self.selector.addItems(names)
            if current and current in self.log_texts:
                self.selector.setCurrentText(current)
            elif names:
                self.selector.setCurrentIndex(0)
            self.selector.blockSignals(False)
            self._force_scroll_to_bottom = True
            self._last_text = None
        self.refresh()

    def _handle_selection_changed(self) -> None:
        self._force_scroll_to_bottom = True
        self.refresh()

    def refresh(self) -> None:
        name = self.selector.currentText()
        text = self.log_texts.get(name)
        if text is None:
            self._last_text = ""
            self.editor.setPlainText("")
            return
        lines = text.splitlines()
        tail = "\n".join(lines[-400:])
        if tail == self._last_text:
            return
        self.editor.setPlainText(tail)
        self._last_text = tail
        QTimer.singleShot(0, self._scroll_to_bottom)
        self._force_scroll_to_bottom = False

    def _scroll_to_bottom(self) -> None:
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.editor.setTextCursor(cursor)
        scrollbar = self.editor.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
