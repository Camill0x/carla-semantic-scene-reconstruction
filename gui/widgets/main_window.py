from typing import Dict

from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget

from gui.config import APP_NAME
from gui.pages.benchmark_page import BenchmarkPage
from gui.pages.carla_page import CarlaPage
from gui.pages.dataset_page import DatasetPage
from gui.pages.home_page import HomePage
from gui.pages.streaming_page import StreamingPage
from gui.pages.training_page import TrainingPage
from gui.process_manager import ProjectProcessManager
from gui.widgets.process_inspector_dialog import ProcessInspectorDialog
from gui.widgets.workflow_window import WorkflowWindow


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.manager = ProjectProcessManager()
        self.workflow_windows: Dict[str, WorkflowWindow] = {}
        self.setWindowTitle(APP_NAME)
        self._build_ui()
        QTimer.singleShot(0, self._resize_to_contents)
        self.refresh_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setSpacing(14)
        self.setCentralWidget(central)

        header = QFrame()
        header.setFrameShape(QFrame.StyledPanel)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 16, 18, 16)
        labels = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setStyleSheet("font-size: 30px; font-weight: 800;")
        subtitle = QLabel("Choose a workflow and open its dedicated control window.")
        subtitle.setStyleSheet("color: #9db0cc;")
        labels.addWidget(title)
        labels.addWidget(subtitle)
        header_layout.addLayout(labels, 1)
        self.inspector_button = QPushButton("Process Inspector")
        self.inspector_button.clicked.connect(self.open_process_inspector)
        self.clear_logs_button = QPushButton("Clear Logs")
        self.clear_logs_button.setProperty("buttonRole", "secondary")
        self.clear_logs_button.clicked.connect(self.clear_logs)
        header_layout.addWidget(self.inspector_button)
        header_layout.addWidget(self.clear_logs_button)
        root.addWidget(header)

        self.home_page = HomePage(self.manager, lambda messages: None, self.open_workflow)
        home_section = self._section("Workflows", self.home_page)
        home_section.setMaximumWidth(820)
        root.addWidget(home_section, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self.exit_button = QPushButton("Exit")
        self.exit_button.setProperty("buttonRole", "danger")
        self.exit_button.clicked.connect(self.close)
        footer.addWidget(self.exit_button)
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

    def _build_workflow_page(self, route: str):
        if route == "carla":
            return CarlaPage(self.manager, self._append_nowhere)
        if route == "dataset":
            return DatasetPage(self.manager, self._append_nowhere)
        if route == "training":
            return TrainingPage(self.manager, self._append_nowhere)
        if route == "benchmark":
            return BenchmarkPage(self.manager, self._append_nowhere)
        if route == "streaming":
            return StreamingPage(self.manager, self._append_nowhere)
        raise KeyError(route)

    def _append_nowhere(self, messages) -> None:
        return None

    def open_workflow(self, route: str) -> None:
        titles = {
            "carla": "Manual Control",
            "dataset": "Dataset Workflow",
            "training": "Training And Evaluation",
            "benchmark": "Offline Benchmark",
            "streaming": "Live Streaming Inference",
        }
        window = self.workflow_windows.get(route)
        if window is None:
            page = self._build_workflow_page(route)
            window = WorkflowWindow(
                route=route,
                title=titles[route],
                page=page,
                manager=self.manager,
                on_closed=self._show_control_center,
            )
            page.append_activity = window.append_activity
            self.workflow_windows[route] = window
        self.hide()
        window.show()
        window.raise_()
        window.activateWindow()
        window.refresh_ui()
        self.refresh_ui()

    def open_process_inspector(self) -> None:
        dialog = ProcessInspectorDialog(
            fetch_rows=self.manager.status_rows,
            on_stop_selected=lambda name: self.manager.stop_process(name),
            on_stop_all=lambda: self.manager.stop_many(self.manager.running_process_names()),
        )
        dialog.exec()
        self.refresh_ui()
        for window in self.workflow_windows.values():
            window.refresh_ui()

    def clear_logs(self) -> None:
        message = QMessageBox(self)
        message.setWindowTitle("Clear Logs")
        message.setMinimumWidth(560)
        message.setText("Clear all GUI-managed log files?")
        message.setInformativeText("This will truncate the current log files in tmp/gui_logs/*")
        clear_button = message.addButton("Clear Logs", QMessageBox.AcceptRole)
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        clear_button.setStyleSheet("background: #16a34a; color: #ffffff;")
        cancel_button.setStyleSheet("background: #d97706; color: #ffffff;")
        message.exec()
        if message.clickedButton() != clear_button:
            return
        self.manager.clear_logs()
        for window in self.workflow_windows.values():
            window.log_viewer.set_logs({})
            window.refresh_ui()

    def refresh_ui(self) -> None:
        return None

    def _show_control_center(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

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
        width = min(max(760, hint.width() + frame_extra.width() + 24), int(available.width() * 0.68))
        height = min(max(580, hint.height() + frame_extra.height() + 24), int(available.height() * 0.82))
        self.resize(width, height)

    def closeEvent(self, event) -> None:
        running = self.manager.running_process_names()
        if not running:
            self.manager.save_state()
            super().closeEvent(event)
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
        self.manager.save_state()
        event.accept()
