from datetime import datetime
from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.config import APP_NAME
from gui.pages.base import WorkflowPage
from gui.process_manager import ProjectProcessManager
from gui.widgets.log_viewer import LogViewer
from gui.widgets.process_inspector_dialog import ProcessInspectorDialog
from gui.widgets.session_summary import SessionSummary


class WorkflowWindow(QMainWindow):
    def __init__(
        self,
        *,
        route: str,
        title: str,
        page: WorkflowPage,
        manager: ProjectProcessManager,
        on_closed: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__()
        self.route = route
        self.page = page
        self.manager = manager
        self.on_closed = on_closed
        self.setWindowTitle(f"{APP_NAME} - {title}")
        self._build_ui(title)
        self._build_timer()
        self.refresh_ui()
        QTimer.singleShot(0, self._resize_to_contents)

    def _build_ui(self, title_text: str) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setSpacing(14)
        self.setCentralWidget(central)

        header = QFrame()
        header.setFrameShape(QFrame.StyledPanel)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 16, 18, 16)
        labels = QVBoxLayout()
        title = QLabel(title_text)
        title.setStyleSheet("font-size: 28px; font-weight: 800;")
        subtitle = QLabel(self.page.window_subtitle())
        subtitle.setStyleSheet("color: #9db0cc;")
        subtitle.setWordWrap(True)
        labels.addWidget(title)
        labels.addWidget(subtitle)
        header_layout.addLayout(labels, 1)
        self.inspector_button = QPushButton("Process Inspector")
        self.inspector_button.clicked.connect(self.open_process_inspector)
        header_layout.addWidget(self.inspector_button)
        root.addWidget(header)

        self.session_summary = SessionSummary(self.page.summary_specs())
        root.addWidget(self._section("Session Summary", self.session_summary))

        body = QSplitter()
        body.setChildrenCollapsible(False)
        root.addWidget(body, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        self.page.setMaximumWidth(920)
        left_scroll.setMinimumWidth(720)
        left_scroll.setWidget(self.page)
        left_scroll.setFrameShape(QFrame.NoFrame)
        body.addWidget(left_scroll)

        right_splitter = QSplitter()
        right_splitter.setOrientation(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        body.addWidget(right_splitter)
        body.setStretchFactor(0, 7)
        body.setStretchFactor(1, 3)

        self.log_viewer = LogViewer()
        logs_section = self._section("Logs", self.log_viewer)
        logs_section.setMinimumWidth(340)
        right_splitter.addWidget(logs_section)
        self.activity_feed = QTextEdit()
        self.activity_feed.setReadOnly(True)
        activity_section = self._section("Activity", self.activity_feed)
        activity_section.setMinimumWidth(340)
        right_splitter.addWidget(activity_section)
        right_splitter.setSizes([180, 120])

        footer = QHBoxLayout()
        footer.addStretch(1)
        self.footer_close_button = QPushButton("Close")
        self.footer_close_button.setProperty("buttonRole", "danger")
        self.footer_close_button.clicked.connect(self.close)
        footer.addWidget(self.footer_close_button)
        root.addLayout(footer)

    def _section(self, title: str, widget: QWidget) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        label = QLabel(title)
        label.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        return frame

    def _build_timer(self) -> None:
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start()

    def append_activity(self, messages) -> None:
        if isinstance(messages, str):
            messages = [messages]
        stamp = datetime.now().strftime("%H:%M:%S")
        for message in messages:
            if message:
                self.activity_feed.append(f"[{stamp}] {message}")
        self.refresh_ui()

    def open_process_inspector(self) -> None:
        dialog = ProcessInspectorDialog(
            fetch_rows=self.manager.status_rows,
            on_stop_selected=lambda name: self.append_activity(self.manager.stop_process(name)),
            on_stop_all=lambda: self.append_activity(self.manager.stop_many(self.manager.running_process_names())),
        )
        dialog.exec()
        self.refresh_ui()

    def refresh_ui(self) -> None:
        logs = {
            name: path
            for name, path in self.manager.available_log_files().items()
            if name in set(self.page.process_names())
        }
        self.log_viewer.set_logs(logs)
        self.session_summary.update_values(self.page.summary_values())
        self.page.refresh()

    def _resize_to_contents(self) -> None:
        central = self.centralWidget()
        if central is None:
            return
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame_extra = self.size() - central.size()
        hint = central.sizeHint()
        cap_width, cap_height = self.page.preferred_window_size()
        width = min(max(900, hint.width() + frame_extra.width() + 32), min(cap_width, int(available.width() * 0.88)))
        height = min(
            max(640, hint.height() + frame_extra.height() + 32),
            min(cap_height, int(available.height() * 0.9)),
        )
        self.resize(width, height)

    def closeEvent(self, event) -> None:
        running = [name for name in self.page.process_names() if name in self.manager.running_process_names()]
        if not running:
            if self.on_closed is not None:
                self.on_closed()
            event.accept()
            return

        message = QMessageBox(self)
        message.setWindowTitle("Active Processes")
        message.setIcon(QMessageBox.Warning)
        message.setText("This workflow still has running processes related to it.")
        message.setInformativeText(
            "\n".join(f"- {name}" for name in running) + "\n\nYou can manage these processes via Process Inspector."
        )
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        close_button = message.addButton("Close", QMessageBox.AcceptRole)
        cancel_button.setStyleSheet("background: #d97706; color: #ffffff;")
        close_button.setStyleSheet("background: #dc2626; color: #ffffff;")
        message.exec()
        clicked = message.clickedButton()
        if clicked == cancel_button:
            event.ignore()
            return
        if self.on_closed is not None:
            self.on_closed()
        event.accept()
