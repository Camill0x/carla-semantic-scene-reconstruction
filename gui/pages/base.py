from PySide6.QtWidgets import QMessageBox, QWidget

from gui.process_manager import ProjectProcessManager
from gui.types import AppendActivity, SummarySpec, SummaryValues


class WorkflowPage(QWidget):
    def __init__(self, manager: ProjectProcessManager, append_activity: AppendActivity) -> None:
        """Initialize the workflow page with a shared process manager and activity sink."""
        super().__init__()
        self.manager = manager
        self.append_activity = append_activity

    def process_names(self) -> list[str]:
        """Return the process names managed by this workflow page."""
        return []

    def refresh(self) -> None:
        """Refresh the workflow page widgets from the latest process state."""
        return None

    def window_subtitle(self) -> str:
        """Return the subtitle shown at the top of the workflow window."""
        return "Dedicated workflow window with process controls and safety checks."

    def summary_specs(self) -> list[SummarySpec]:
        """Return the summary-card definitions shown for this workflow."""
        return [("running", "Active Processes")]

    def summary_values(self) -> SummaryValues:
        """Return the current values displayed in the workflow summary cards."""
        return {
            "running": str(len([name for name in self.process_names() if name in self.manager.running_process_names()]))
        }

    def preferred_window_size(self) -> tuple[int, int]:
        """Return the preferred size for the workflow window."""
        return (1060, 720)

    def notify_error(self, title: str, text: str) -> None:
        """Show a warning dialog for an invalid workflow action or configuration."""
        QMessageBox.warning(self, title, text)
