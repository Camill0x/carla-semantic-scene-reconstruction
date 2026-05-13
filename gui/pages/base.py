from PySide6.QtWidgets import QMessageBox, QWidget

from gui.process_manager import ProjectProcessManager
from gui.types import AppendActivity, SummarySpec, SummaryValues


class WorkflowPage(QWidget):
    def __init__(self, manager: ProjectProcessManager, append_activity: AppendActivity) -> None:
        super().__init__()
        self.manager = manager
        self.append_activity = append_activity

    def process_names(self) -> list[str]:
        return []

    def refresh(self) -> None:
        return None

    def window_subtitle(self) -> str:
        return "Dedicated workflow window with process controls and safety checks."

    def summary_specs(self) -> list[SummarySpec]:
        return [("running", "Active Processes")]

    def summary_values(self) -> SummaryValues:
        return {
            "running": str(len([name for name in self.process_names() if name in self.manager.running_process_names()]))
        }

    def preferred_window_size(self) -> tuple[int, int]:
        return (1060, 720)

    def notify_error(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)
