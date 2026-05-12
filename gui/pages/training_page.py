from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget, QVBoxLayout as InnerLayout

from gui.catalog import (
    LANEDET_MAIN_FLAGS,
    LANEDET_PREPARE_FLAGS,
    OPENPCDET_PREPARE_FLAGS,
    OPENPCDET_TEST_FLAGS,
    OPENPCDET_TRAIN_FLAGS,
)
from gui.pages.base import WorkflowPage
from gui.widgets.process_panel import ProcessPanel


class TrainingPage(WorkflowPage):
    def __init__(self, manager, append_activity) -> None:
        super().__init__(manager, append_activity)
        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._openpcdet_tab(), "OpenPCDet")
        tabs.addTab(self._lanedet_tab(), "LaneDet")
        layout.addWidget(tabs)

    def window_subtitle(self) -> str:
        return "Prepare datasets, launch training jobs and run checkpoint evaluation for both OpenPCDet and LaneDet."

    def preferred_window_size(self):
        return (1020, 780)

    def summary_specs(self):
        return [
            ("running", "Active Processes"),
            ("prepare", "Preparation Jobs"),
            ("train", "Training Jobs"),
            ("eval", "Evaluation Jobs"),
        ]

    def summary_values(self):
        running = self.manager.running_process_names()
        prepare = sum(1 for name in ["openpcdet_prepare_dataset", "lanedet_prepare_dataset"] if name in running)
        train = sum(1 for name in ["openpcdet_train", "lanedet_main"] if name in running)
        eval_count = 1 if "openpcdet_test" in running else 0
        return {
            "running": f"{len([name for name in self.process_names() if name in running])} / {len(self.process_names())}",
            "prepare": str(prepare),
            "train": str(train),
            "eval": str(eval_count),
        }

    def process_names(self):
        return [
            "openpcdet_prepare_dataset",
            "openpcdet_train",
            "openpcdet_test",
            "lanedet_prepare_dataset",
            "lanedet_main",
        ]

    def _panel(self, process_name, title, description, flags, allow_extra_args):
        return ProcessPanel(
            title=title,
            description=description,
            flags=flags,
            on_start=lambda args, name=process_name: self._start(name, args),
            on_stop=lambda name=process_name: self.append_activity(self.manager.stop_process(name)),
            on_restart=lambda args, name=process_name: self.append_activity(
                self.manager.restart_process(name, args=args)
            ),
            allow_extra_args=allow_extra_args,
            initial_args=self.manager.args_for(process_name),
        )

    def _openpcdet_tab(self) -> QWidget:
        tab = QWidget()
        layout = InnerLayout(tab)
        self.openpcdet_prepare_panel = self._panel(
            "openpcdet_prepare_dataset",
            "Prepare OpenPCDet Dataset",
            "Build the prepared dataset metadata from collected raw runs.",
            OPENPCDET_PREPARE_FLAGS,
            False,
        )
        self.openpcdet_train_panel = self._panel(
            "openpcdet_train",
            "Train OpenPCDet",
            "Launch training with either a preset or an explicit config file.",
            OPENPCDET_TRAIN_FLAGS,
            True,
        )
        self.openpcdet_test_panel = self._panel(
            "openpcdet_test",
            "Test OpenPCDet",
            "Evaluate a checkpoint on the held-out test split.",
            OPENPCDET_TEST_FLAGS,
            True,
        )
        layout.addWidget(self.openpcdet_prepare_panel)
        layout.addWidget(self.openpcdet_train_panel)
        layout.addWidget(self.openpcdet_test_panel)
        layout.addStretch(1)
        return tab

    def _lanedet_tab(self) -> QWidget:
        tab = QWidget()
        layout = InnerLayout(tab)
        self.lanedet_prepare_panel = self._panel(
            "lanedet_prepare_dataset",
            "Prepare LaneDet Dataset",
            "Convert collected runs into the TuSimple-like dataset expected by LaneDet.",
            LANEDET_PREPARE_FLAGS,
            False,
        )
        self.lanedet_main_panel = self._panel(
            "lanedet_main",
            "Run LaneDet",
            "Use the same panel for training or validation, depending on the selected flags.",
            LANEDET_MAIN_FLAGS,
            True,
        )
        layout.addWidget(self.lanedet_prepare_panel)
        layout.addWidget(self.lanedet_main_panel)
        layout.addStretch(1)
        return tab

    def _start(self, process_name: str, args):
        panel = {
            "openpcdet_prepare_dataset": self.openpcdet_prepare_panel,
            "openpcdet_train": self.openpcdet_train_panel,
            "openpcdet_test": self.openpcdet_test_panel,
            "lanedet_prepare_dataset": self.lanedet_prepare_panel,
            "lanedet_main": self.lanedet_main_panel,
        }[process_name]
        error = panel.validation_error()
        if error:
            self.notify_error("Invalid Training Configuration", error)
            return
        self.append_activity(self.manager.start_process(process_name, args=args))

    def refresh(self) -> None:
        rows = {row["name"]: row for row in self.manager.status_rows()}
        self.openpcdet_prepare_panel.set_status(rows["openpcdet_prepare_dataset"]["status"])
        self.openpcdet_train_panel.set_status(rows["openpcdet_train"]["status"])
        self.openpcdet_test_panel.set_status(rows["openpcdet_test"]["status"])
        self.lanedet_prepare_panel.set_status(rows["lanedet_prepare_dataset"]["status"])
        self.lanedet_main_panel.set_status(rows["lanedet_main"]["status"])
