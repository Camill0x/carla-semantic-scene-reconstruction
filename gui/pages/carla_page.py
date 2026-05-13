from PySide6.QtWidgets import QGridLayout, QVBoxLayout

from gui.catalog import MANUAL_CONTROL_FLAGS, TRAFFIC_FLAGS
from gui.pages.base import WorkflowPage
from gui.process_manager import ProjectProcessManager
from gui.types import AppendActivity, ArgsList, SummarySpec, SummaryValues
from gui.widgets.process_panel import ProcessPanel


class CarlaPage(WorkflowPage):
    def __init__(self, manager: ProjectProcessManager, append_activity: AppendActivity) -> None:
        """Build the CARLA workflow page and its simulator control panels."""
        super().__init__(manager, append_activity)
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        self.server_panel = ProcessPanel(
            title="CARLA Server",
            description="Start the simulator process. Everything else in this window depends on it.",
            flags=[],
            on_start=lambda args: self.append_activity(self.manager.start_process("server", args=args)),
            on_stop=self._stop_server,
            on_restart=self._restart_server,
            allow_extra_args=True,
            initial_args=self.manager.args_for("server"),
        )
        self.manual_panel = ProcessPanel(
            title="Manual Control",
            description="Spawn and control the hero vehicle. Sync mode is always enabled.",
            flags=MANUAL_CONTROL_FLAGS,
            on_start=self._start_manual,
            on_stop=self._stop_manual,
            on_restart=self._restart_manual,
            allow_extra_args=True,
            initial_args=self.manager.args_for("manual_control"),
        )
        self.traffic_panel = ProcessPanel(
            title="Generate Traffic",
            description="Populate the scene with NPC vehicles and pedestrians after the hero vehicle is running.",
            flags=TRAFFIC_FLAGS,
            on_start=self._start_traffic,
            on_stop=self._stop_traffic,
            on_restart=self._restart_traffic,
            allow_extra_args=True,
            initial_args=self.manager.args_for("traffic"),
        )
        grid.addWidget(self.server_panel, 0, 0)
        grid.addWidget(self.manual_panel, 0, 1)
        grid.addWidget(self.traffic_panel, 1, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch(1)

    def window_subtitle(self) -> str:
        """Return the subtitle shown for the CARLA workflow window."""
        return "Quick simulator workflow for checking that CARLA, ego spawning and traffic generation behave correctly."

    def preferred_window_size(self) -> tuple[int, int]:
        """Return the preferred size for the CARLA workflow window."""
        return (940, 700)

    def summary_specs(self) -> list[SummarySpec]:
        """Return the summary-card definitions for the CARLA workflow."""
        return [
            ("running", "Active Processes"),
            ("server", "Server"),
            ("ego", "Ego Control"),
            ("traffic", "Traffic"),
        ]

    def summary_values(self) -> SummaryValues:
        """Return the current summary values for the CARLA workflow."""
        running = self.manager.running_process_names()
        return {
            "running": f"{len([name for name in self.process_names() if name in running])} / {len(self.process_names())}",
            "server": "Running" if "server" in running else "Stopped",
            "ego": "Running" if "manual_control" in running else "Stopped",
            "traffic": "Running" if "traffic" in running else "Stopped",
        }

    def process_names(self) -> list[str]:
        """Return the process names managed by the CARLA workflow."""
        return ["server", "manual_control", "traffic"]

    def _start_manual(self, args: ArgsList) -> None:
        """Validate prerequisites and start the manual-control process."""
        if "server" not in self.manager.running_process_names():
            self.notify_error("Server Required", "Start the CARLA server before launching manual control.")
            return
        error = self.manual_panel.validation_error()
        if error:
            self.notify_error("Invalid Manual Control Config", error)
            return
        self.append_activity(self.manager.start_process("manual_control", args=args))

    def _start_traffic(self, args: ArgsList) -> None:
        """Validate prerequisites and start the traffic-generation process."""
        running = self.manager.running_process_names()
        if "manual_control" not in running:
            self.notify_error(
                "Manual Control Required", "Traffic generation requires an active manual_control process."
            )
            return
        self.append_activity(self.manager.start_process("traffic", args=args))

    def _stop_server(self) -> None:
        """Stop the server together with dependent CARLA workflow processes."""
        self.append_activity(self.manager.stop_many(["traffic", "manual_control", "server"]))

    def _restart_server(self, args: ArgsList) -> None:
        """Restart the server after stopping dependent CARLA workflow processes."""
        self.append_activity(self.manager.stop_many(["traffic", "manual_control", "server"]))
        self.append_activity(self.manager.start_process("server", args=args))

    def _stop_manual(self) -> None:
        """Stop manual control together with dependent traffic generation."""
        self.append_activity(self.manager.stop_many(["traffic", "manual_control"]))

    def _restart_manual(self, args: ArgsList) -> None:
        """Restart manual control after stopping dependent traffic generation."""
        self.append_activity(self.manager.stop_many(["traffic", "manual_control"]))
        self._start_manual(args)

    def _stop_traffic(self) -> None:
        """Stop the traffic-generation process."""
        self.append_activity(self.manager.stop_process("traffic"))

    def _restart_traffic(self, args: ArgsList) -> None:
        """Restart the traffic-generation process."""
        self.append_activity(self.manager.stop_process("traffic"))
        self._start_traffic(args)

    def refresh(self) -> None:
        """Refresh CARLA panel statuses from the process manager."""
        rows = {row["name"]: row for row in self.manager.status_rows()}
        self.server_panel.set_status(rows["server"]["status"])
        self.manual_panel.set_status(rows["manual_control"]["status"])
        self.traffic_panel.set_status(rows["traffic"]["status"])
