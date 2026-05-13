from typing import Callable, List

from PySide6.QtWidgets import QMessageBox, QWidget

from gui.process_manager import ProjectProcessManager


class WorkflowPage(QWidget):
    def __init__(self, manager: ProjectProcessManager, append_activity: Callable[[object], None]) -> None:
        super().__init__()
        self.manager = manager
        self.append_activity = append_activity

    def process_names(self) -> List[str]:
        return []

    def refresh(self) -> None:
        return None

    def window_subtitle(self) -> str:
        return "Dedicated workflow window with process controls and safety checks."

    def summary_specs(self):
        return [("running", "Active Processes")]

    def summary_values(self):
        return {
            "running": str(len([name for name in self.process_names() if name in self.manager.running_process_names()]))
        }

    def preferred_window_size(self):
        return (1060, 720)

    def notify_error(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)
