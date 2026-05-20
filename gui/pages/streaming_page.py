from PySide6.QtWidgets import QGridLayout, QVBoxLayout

from gui.catalog import (
    MANUAL_CONTROL_FLAGS,
    STREAMING_LANEDET_FLAGS,
    STREAMING_OPENPCDET_FLAGS,
    STREAMING_PRODUCER_FLAGS,
    TRAFFIC_FLAGS,
)
from gui.pages.base import WorkflowPage
from gui.process_manager import ProjectProcessManager
from gui.types import AppendActivity, ArgsList, SummarySpec, SummaryValues
from gui.widgets.process_panel import ProcessPanel


class StreamingPage(WorkflowPage):
    def __init__(self, manager: ProjectProcessManager, append_activity: AppendActivity) -> None:
        """Build the live-streaming workflow page and its process control panels."""
        super().__init__(manager, append_activity)
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        self.server_panel = ProcessPanel(
            title="CARLA Server",
            description="Start the simulator process before manual driving and live streaming inference.",
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
            description="Populate the scene with NPC vehicles and pedestrians.",
            flags=TRAFFIC_FLAGS,
            on_start=self._start_traffic,
            on_stop=self._stop_traffic,
            on_restart=self._restart_traffic,
            allow_extra_args=True,
            initial_args=self.manager.args_for("traffic"),
        )
        self.core_panel = ProcessPanel(
            title="Inference Core",
            description="Start the producer, aggregator and visualizer that form the core live inference path.",
            flags=STREAMING_PRODUCER_FLAGS,
            on_start=self._start_core,
            on_stop=self._stop_core,
            on_restart=self._restart_core,
            allow_extra_args=False,
            initial_args=(
                self.manager.args_for("stream_producer")
                + self.manager.args_for("stream_aggregator")
                + self.manager.args_for("stream_visualizer")
            ),
        )
        self.openpcdet_panel = ProcessPanel(
            title="OpenPCDet Detector",
            description="Attach the 3D object detector to the live streaming pipeline.",
            flags=STREAMING_OPENPCDET_FLAGS,
            on_start=lambda args: self._start_detector("stream_openpcdet", args),
            on_stop=lambda: self.append_activity(self.manager.stop_process("stream_openpcdet")),
            on_restart=lambda args: self.append_activity(self.manager.restart_process("stream_openpcdet", args=args)),
            allow_extra_args=False,
            initial_args=self.manager.args_for("stream_openpcdet"),
        )
        self.lanedet_panel = ProcessPanel(
            title="LaneDet Detector",
            description="Attach the lane detector to the same live streaming pipeline.",
            flags=STREAMING_LANEDET_FLAGS,
            on_start=lambda args: self._start_detector("stream_lanedet", args),
            on_stop=lambda: self.append_activity(self.manager.stop_process("stream_lanedet")),
            on_restart=lambda args: self.append_activity(self.manager.restart_process("stream_lanedet", args=args)),
            allow_extra_args=False,
            initial_args=self.manager.args_for("stream_lanedet"),
        )
        grid.addWidget(self.server_panel, 0, 0)
        grid.addWidget(self.manual_panel, 0, 1)
        grid.addWidget(self.traffic_panel, 1, 0, 1, 2)
        grid.addWidget(self.core_panel, 2, 0, 1, 2)
        grid.addWidget(self.openpcdet_panel, 3, 0)
        grid.addWidget(self.lanedet_panel, 3, 1)
        layout.addLayout(grid)
        layout.addStretch(1)

    def window_subtitle(self) -> str:
        """Return the subtitle shown for the live-streaming workflow window."""
        return "Live inference workflow where raw camera and LiDAR stay local and only detections reach the viewer."

    def preferred_window_size(self) -> tuple[int, int]:
        """Return the preferred size for the live-streaming workflow window."""
        return (1060, 760)

    def summary_specs(self) -> list[SummarySpec]:
        """Return the summary-card definitions for the live-streaming workflow."""
        return [
            ("running", "Active Processes"),
            ("server", "Server"),
            ("ego", "Manual Control"),
            ("traffic", "Traffic"),
            ("core", "Inference Core"),
            ("detectors", "Active Detectors"),
        ]

    def summary_values(self) -> SummaryValues:
        """Return the current summary values for the live-streaming workflow."""
        running = self.manager.running_process_names()
        detector_count = sum(1 for name in ["stream_openpcdet", "stream_lanedet"] if name in running)
        core_ok = all(name in running for name in ["stream_producer", "stream_aggregator", "stream_visualizer"])
        return {
            "running": f"{len([name for name in self.process_names() if name in running])} / {len(self.process_names())}",
            "server": "Running" if "server" in running else "Stopped",
            "ego": "Running" if "manual_control" in running else "Stopped",
            "traffic": "Running" if "traffic" in running else "Stopped",
            "core": "Ready" if core_ok else "Partial",
            "detectors": str(detector_count),
        }

    def process_names(self) -> list[str]:
        """Return the process names managed by the live-streaming workflow."""
        return [
            "server",
            "manual_control",
            "traffic",
            "stream_producer",
            "stream_aggregator",
            "stream_visualizer",
            "stream_openpcdet",
            "stream_lanedet",
        ]

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
        if "manual_control" not in self.manager.running_process_names():
            self.notify_error(
                "Manual Control Required", "Traffic generation requires an active manual_control process."
            )
            return
        self.append_activity(self.manager.start_process("traffic", args=args))

    def _start_core(self, args: ArgsList) -> None:
        """Validate prerequisites and start the producer, aggregator, and visualizer."""
        running = self.manager.running_process_names()
        if "manual_control" not in running:
            self.notify_error(
                "Manual Control Required", "Start manual control before launching the streaming inference core."
            )
            return
        error = self.core_panel.validation_error()
        if error:
            self.notify_error("Invalid Producer Configuration", error)
            return
        producer_args, aggregator_args, visualizer_args = self._core_args()
        messages = [
            self.manager.start_process("stream_producer", args=producer_args),
            self.manager.start_process("stream_aggregator", args=aggregator_args),
            self.manager.start_process("stream_visualizer", args=visualizer_args),
        ]
        self.append_activity(messages)

    def _stop_core(self) -> None:
        """Stop the streaming core together with any attached detectors."""
        self.append_activity(
            self.manager.stop_many(
                ["stream_lanedet", "stream_openpcdet", "stream_visualizer", "stream_aggregator", "stream_producer"]
            )
        )

    def _restart_core(self, args: ArgsList) -> None:
        """Restart the streaming core after stopping its active processes."""
        self._stop_core()
        self._start_core(args)

    def _core_args(self) -> tuple[ArgsList, ArgsList, ArgsList]:
        """Split the shared core form values into producer, aggregator, and visualizer arguments."""
        values = self.core_panel.form.values()
        producer_args: ArgsList = []
        aggregator_args: ArgsList = []
        visualizer_args: ArgsList = []

        every_nth = str(values.get("every_nth", "")).strip()
        if every_nth:
            producer_args.extend(["--every-nth", every_nth])

        if values.get("show_grid"):
            visualizer_args.append("--show-grid")

        if values.get("verbose"):
            producer_args.append("--verbose")
            aggregator_args.append("--verbose")
            visualizer_args.append("--verbose")

        return producer_args, aggregator_args, visualizer_args

    def _start_detector(self, name: str, args: ArgsList) -> None:
        """Validate prerequisites and start one live detector process."""
        running = self.manager.running_process_names()
        if not all(component in running for component in ["stream_producer", "stream_aggregator", "stream_visualizer"]):
            self.notify_error(
                "Inference Core Required", "Start the full streaming inference core before adding detectors."
            )
            return
        panel = self.openpcdet_panel if name == "stream_openpcdet" else self.lanedet_panel
        error = panel.validation_error()
        if error:
            self.notify_error("Invalid Detector Configuration", error)
            return
        self.append_activity(self.manager.start_process(name, args=args))

    def _stop_server(self) -> None:
        """Stop the server together with dependent streaming workflow processes."""
        self.append_activity(
            self.manager.stop_many(
                [
                    "stream_lanedet",
                    "stream_openpcdet",
                    "stream_visualizer",
                    "stream_aggregator",
                    "stream_producer",
                    "traffic",
                    "manual_control",
                    "server",
                ]
            )
        )

    def _restart_server(self, args: ArgsList) -> None:
        """Restart the server after stopping dependent streaming workflow processes."""
        self._stop_server()
        self.append_activity(self.manager.start_process("server", args=args))

    def _stop_manual(self) -> None:
        """Stop manual control together with dependent streaming workflow processes."""
        self.append_activity(
            self.manager.stop_many(
                [
                    "stream_lanedet",
                    "stream_openpcdet",
                    "stream_visualizer",
                    "stream_aggregator",
                    "stream_producer",
                    "traffic",
                    "manual_control",
                ]
            )
        )

    def _restart_manual(self, args: ArgsList) -> None:
        """Restart manual control after stopping dependent streaming workflow processes."""
        self._stop_manual()
        self._start_manual(args)

    def _stop_traffic(self) -> None:
        """Stop the traffic-generation process."""
        self.append_activity(self.manager.stop_process("traffic"))

    def _restart_traffic(self, args: ArgsList) -> None:
        """Restart the traffic-generation process."""
        self.append_activity(self.manager.stop_process("traffic"))
        self._start_traffic(args)

    def refresh(self) -> None:
        """Refresh streaming workflow panel statuses from the process manager."""
        rows = {row["name"]: row for row in self.manager.status_rows()}
        core_status = (
            "Running"
            if rows["stream_producer"]["status"] == "Running"
            and rows["stream_aggregator"]["status"] == "Running"
            and rows["stream_visualizer"]["status"] == "Running"
            else "Stopped"
        )
        self.server_panel.set_status(rows["server"]["status"])
        self.manual_panel.set_status(rows["manual_control"]["status"])
        self.traffic_panel.set_status(rows["traffic"]["status"])
        self.core_panel.set_status(core_status)
        self.openpcdet_panel.set_status(rows["stream_openpcdet"]["status"])
        self.lanedet_panel.set_status(rows["stream_lanedet"]["status"])
