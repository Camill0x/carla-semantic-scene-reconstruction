from functools import partial

from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout

from gui.pages.base import WorkflowPage
from gui.process_manager import ProjectProcessManager
from gui.types import AppendActivity, OpenWorkflow
from gui.widgets.navigation_card import NavigationCard


class HomePage(WorkflowPage):
    def __init__(
        self,
        manager: ProjectProcessManager,
        append_activity: AppendActivity,
        open_workflow: OpenWorkflow,
    ) -> None:
        """Build the home page with navigation cards for each major workflow."""
        super().__init__(manager, append_activity)
        layout = QVBoxLayout(self)
        self.setMaximumWidth(760)
        title = QLabel("Project Control Center")
        title.setStyleSheet("font-size: 30px; font-weight: 800;")
        subtitle = QLabel(
            "Use this panel to move between CARLA driving, dataset collection, model training, benchmarking and the live streaming inference pipeline."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #9db0cc; font-size: 15px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        cards = [
            (
                "Manual Control",
                "Start the CARLA server, spawn ego driving and generate traffic. Good for quick simulator checks.",
                "Open Manual Control",
                "carla",
            ),
            (
                "Dataset Collection",
                "Drive, generate traffic, collect synchronized frames and open the saved run in the dataset viewer.",
                "Open Dataset Workflow",
                "dataset",
            ),
            (
                "Training",
                "Prepare datasets and launch OpenPCDet or LaneDet training and evaluation jobs from one place.",
                "Open Training",
                "training",
            ),
            (
                "Benchmark",
                "Run offline benchmarks and inspect predicted objects and lanes against saved runs.",
                "Open Benchmark",
                "benchmark",
            ),
            (
                "Live Streaming",
                "Control the live producer, aggregator, visualizer and both detectors for low-bandwidth inference demos.",
                "Open Streaming",
                "streaming",
            ),
        ]
        for index, (title_text, description, button, route) in enumerate(cards):
            card = NavigationCard(
                title=title_text,
                description=description,
                button_text=button,
                on_click=partial(open_workflow, route),
            )
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)
        layout.addStretch(1)
