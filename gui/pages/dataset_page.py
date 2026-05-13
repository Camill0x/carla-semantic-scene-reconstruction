from PySide6.QtWidgets import QGridLayout, QVBoxLayout

from gui.catalog import DATASET_COLLECT_FLAGS, DATASET_VIEW_FLAGS, MANUAL_CONTROL_FLAGS, TRAFFIC_FLAGS
from gui.pages.base import WorkflowPage
from gui.process_manager import ProjectProcessManager
from gui.types import AppendActivity, ArgsList, SummarySpec, SummaryValues
from gui.widgets.process_panel import ProcessPanel


class DatasetPage(WorkflowPage):
    def __init__(self, manager: ProjectProcessManager, append_activity: AppendActivity) -> None:
        """Build the dataset workflow page and its collection and viewer panels."""
        super().__init__(manager, append_activity)
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        self.server_panel = ProcessPanel(
            title="CARLA Server",
            description="Keep the simulator running while you drive, collect frames and inspect saved runs.",
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
            description="Optional NPC traffic for richer scenes while collecting the run.",
            flags=TRAFFIC_FLAGS,
            on_start=self._start_traffic,
            on_stop=self._stop_traffic,
            on_restart=self._restart_traffic,
            allow_extra_args=True,
            initial_args=self.manager.args_for("traffic"),
        )
        self.collect_panel = ProcessPanel(
            title="Collect Dataset",
            description="Record synchronized front camera, LiDAR and metadata into a fresh `run_XXXX` directory.",
            flags=DATASET_COLLECT_FLAGS,
            on_start=self._start_collect,
            on_stop=lambda: self.append_activity(self.manager.stop_process("dataset_collect")),
            on_restart=lambda args: self.append_activity(self.manager.restart_process("dataset_collect", args=args)),
            allow_extra_args=False,
            initial_args=self.manager.args_for("dataset_collect"),
        )
        self.show_panel = ProcessPanel(
            title="Show Dataset",
            description="Replay a saved dataset run in the offline viewer to inspect what was captured.",
            flags=DATASET_VIEW_FLAGS,
            on_start=self._start_show,
            on_stop=lambda: self.append_activity(self.manager.stop_process("dataset_view")),
            on_restart=lambda args: self.append_activity(self.manager.restart_process("dataset_view", args=args)),
            allow_extra_args=False,
            initial_args=self.manager.args_for("dataset_view"),
        )

        grid.addWidget(self.server_panel, 0, 0)
        grid.addWidget(self.manual_panel, 0, 1)
        grid.addWidget(self.traffic_panel, 1, 0)
        grid.addWidget(self.collect_panel, 1, 1)
        grid.addWidget(self.show_panel, 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch(1)

    def window_subtitle(self) -> str:
        """Return the subtitle shown for the dataset workflow window."""
        return (
            "End-to-end workflow for driving in CARLA, collecting multimodal frames and reviewing saved runs afterward."
        )

    def preferred_window_size(self) -> tuple[int, int]:
        """Return the preferred size for the dataset workflow window."""
        return (1040, 760)

    def summary_specs(self) -> list[SummarySpec]:
        """Return the summary-card definitions for the dataset workflow."""
        return [
            ("running", "Active Processes"),
            ("drive", "Driving"),
            ("collect", "Collecting"),
            ("viewer", "Viewer"),
        ]

    def summary_values(self) -> SummaryValues:
        """Return the current summary values for the dataset workflow."""
        running = self.manager.running_process_names()
        return {
            "running": f"{len([name for name in self.process_names() if name in running])} / {len(self.process_names())}",
            "drive": "Ready" if "manual_control" in running else "Stopped",
            "collect": "Running" if "dataset_collect" in running else "Idle",
            "viewer": "Open" if "dataset_view" in running else "Closed",
        }

    def process_names(self) -> list[str]:
        """Return the process names managed by the dataset workflow."""
        return ["server", "manual_control", "traffic", "dataset_collect", "dataset_view"]

    def _start_manual(self, args: ArgsList) -> None:
        """Validate prerequisites and start the manual-control process."""
        if "server" not in self.manager.running_process_names():
            self.notify_error("Server Required", "Start the CARLA server before manual control.")
            return
        error = self.manual_panel.validation_error()
        if error:
            self.notify_error("Invalid Manual Control Config", error)
            return
        self.append_activity(self.manager.start_process("manual_control", args=args))

    def _start_traffic(self, args: ArgsList) -> None:
        """Validate prerequisites and start the traffic-generation process."""
        if "manual_control" not in self.manager.running_process_names():
            self.notify_error("Manual Control Required", "Traffic generation requires manual control.")
            return
        self.append_activity(self.manager.start_process("traffic", args=args))

    def _start_collect(self, args: ArgsList) -> None:
        """Validate prerequisites and start dataset collection."""
        if "manual_control" not in self.manager.running_process_names():
            self.notify_error("Manual Control Required", "Drive the ego vehicle before collecting a dataset.")
            return
        self.append_activity(self.manager.start_process("dataset_collect", args=args))

    def _start_show(self, args: ArgsList) -> None:
        """Validate the viewer configuration and start dataset replay."""
        error = self.show_panel.validation_error()
        if error:
            self.notify_error("Invalid Dataset Viewer Config", error)
            return
        self.append_activity(self.manager.start_process("dataset_view", args=args))

    def _stop_server(self) -> None:
        """Stop the server together with dependent dataset workflow processes."""
        self.append_activity(
            self.manager.stop_many(["dataset_view", "dataset_collect", "traffic", "manual_control", "server"])
        )

    def _restart_server(self, args: ArgsList) -> None:
        """Restart the server after stopping dependent dataset workflow processes."""
        self.append_activity(
            self.manager.stop_many(["dataset_view", "dataset_collect", "traffic", "manual_control", "server"])
        )
        self.append_activity(self.manager.start_process("server", args=args))

    def _stop_manual(self) -> None:
        """Stop manual control together with dependent dataset workflow processes."""
        self.append_activity(self.manager.stop_many(["dataset_view", "dataset_collect", "traffic", "manual_control"]))

    def _restart_manual(self, args: ArgsList) -> None:
        """Restart manual control after stopping dependent dataset workflow processes."""
        self.append_activity(self.manager.stop_many(["dataset_view", "dataset_collect", "traffic", "manual_control"]))
        self._start_manual(args)

    def _stop_traffic(self) -> None:
        """Stop the traffic-generation process."""
        self.append_activity(self.manager.stop_process("traffic"))

    def _restart_traffic(self, args: ArgsList) -> None:
        """Restart the traffic-generation process."""
        self.append_activity(self.manager.stop_process("traffic"))
        self._start_traffic(args)

    def refresh(self) -> None:
        """Refresh dataset workflow panel statuses from the process manager."""
        rows = {row["name"]: row for row in self.manager.status_rows()}
        self.server_panel.set_status(rows["server"]["status"])
        self.manual_panel.set_status(rows["manual_control"]["status"])
        self.traffic_panel.set_status(rows["traffic"]["status"])
        self.collect_panel.set_status(rows["dataset_collect"]["status"])
        self.show_panel.set_status(rows["dataset_view"]["status"])
