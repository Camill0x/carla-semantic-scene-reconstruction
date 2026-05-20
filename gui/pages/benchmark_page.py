from PySide6.QtWidgets import QGridLayout, QVBoxLayout

from gui.catalog import BENCHMARK_LANEDET_FLAGS, BENCHMARK_OPENPCDET_FLAGS, BENCHMARK_VIEW_FLAGS
from gui.pages.base import WorkflowPage
from gui.process_manager import ProjectProcessManager
from gui.types import AppendActivity, ArgsList, SummarySpec, SummaryValues
from gui.widgets.process_panel import ProcessPanel


class BenchmarkPage(WorkflowPage):
    def __init__(self, manager: ProjectProcessManager, append_activity: AppendActivity) -> None:
        """Build the benchmark workflow page and its process control panels."""
        super().__init__(manager, append_activity)
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        self.openpcdet_panel = ProcessPanel(
            title="Benchmark OpenPCDet",
            description="Measure OpenPCDet speed and prediction volume on a saved raw run.",
            flags=BENCHMARK_OPENPCDET_FLAGS,
            on_start=lambda args: self._start("benchmark_openpcdet", args),
            on_stop=lambda: self.append_activity(self.manager.stop_process("benchmark_openpcdet")),
            on_restart=lambda args: self.append_activity(
                self.manager.restart_process("benchmark_openpcdet", args=args)
            ),
            allow_extra_args=False,
            initial_args=self.manager.args_for("benchmark_openpcdet"),
        )
        self.lanedet_panel = ProcessPanel(
            title="Benchmark LaneDet",
            description="Measure LaneDet speed and lane output on the same kind of saved run.",
            flags=BENCHMARK_LANEDET_FLAGS,
            on_start=lambda args: self._start("benchmark_lanedet", args),
            on_stop=lambda: self.append_activity(self.manager.stop_process("benchmark_lanedet")),
            on_restart=lambda args: self.append_activity(self.manager.restart_process("benchmark_lanedet", args=args)),
            allow_extra_args=False,
            initial_args=self.manager.args_for("benchmark_lanedet"),
        )
        self.view_panel = ProcessPanel(
            title="View Predictions",
            description="Inspect saved benchmark predictions in Rerun after the benchmark finishes.",
            flags=BENCHMARK_VIEW_FLAGS,
            on_start=lambda args: self._start("benchmark_view_predictions", args),
            on_stop=lambda: self.append_activity(self.manager.stop_process("benchmark_view_predictions")),
            on_restart=lambda args: self.append_activity(
                self.manager.restart_process("benchmark_view_predictions", args=args)
            ),
            allow_extra_args=False,
            initial_args=self.manager.args_for("benchmark_view_predictions"),
        )
        grid.addWidget(self.openpcdet_panel, 0, 0)
        grid.addWidget(self.lanedet_panel, 0, 1)
        grid.addWidget(self.view_panel, 1, 0, 1, 2)
        layout.addLayout(grid)
        layout.addStretch(1)

    def window_subtitle(self) -> str:
        """Return the subtitle shown for the benchmark workflow window."""
        return "Offline measurement workflow for comparing model throughput and then browsing saved predictions."

    def preferred_window_size(self) -> tuple[int, int]:
        """Return the preferred size for the benchmark workflow window."""
        return (980, 720)

    def summary_specs(self) -> list[SummarySpec]:
        """Return the summary-card definitions for the benchmark workflow."""
        return [
            ("running", "Active Processes"),
            ("pcdet", "OpenPCDet"),
            ("lanedet", "LaneDet"),
            ("viewer", "Prediction Viewer"),
        ]

    def summary_values(self) -> SummaryValues:
        """Return the current summary values for the benchmark workflow."""
        running = self.manager.running_process_names()
        return {
            "running": f"{len([name for name in self.process_names() if name in running])} / {len(self.process_names())}",
            "pcdet": "Running" if "benchmark_openpcdet" in running else "Idle",
            "lanedet": "Running" if "benchmark_lanedet" in running else "Idle",
            "viewer": "Open" if "benchmark_view_predictions" in running else "Closed",
        }

    def process_names(self) -> list[str]:
        """Return the process names managed by the benchmark workflow."""
        return ["benchmark_openpcdet", "benchmark_lanedet", "benchmark_view_predictions"]

    def _start(self, name: str, args: ArgsList) -> None:
        """Validate the selected benchmark panel and start its process."""
        panel = {
            "benchmark_openpcdet": self.openpcdet_panel,
            "benchmark_lanedet": self.lanedet_panel,
            "benchmark_view_predictions": self.view_panel,
        }[name]
        error = panel.validation_error()
        if error:
            self.notify_error("Invalid Benchmark Configuration", error)
            return
        self.append_activity(self.manager.start_process(name, args=args))

    def refresh(self) -> None:
        """Refresh benchmark panel statuses from the process manager."""
        rows = {row["name"]: row for row in self.manager.status_rows()}
        self.openpcdet_panel.set_status(rows["benchmark_openpcdet"]["status"])
        self.lanedet_panel.set_status(rows["benchmark_lanedet"]["status"])
        self.view_panel.set_status(rows["benchmark_view_predictions"]["status"])
