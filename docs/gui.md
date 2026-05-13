# GUI

This document describes the Project Control Center used to orchestrate common project workflows from one place. The GUI does not replace the Python entrypoints in `tools/` — instead, it provides a thin process-management layer on top of the documented CLI workflows.

Before using the GUI, complete the setup in [installation.md](installation.md). In practice, you need at least the `carla_app` environment. Some workflow windows also require the `openpcdet` or `lanedet` environments because they launch model-specific commands in those Conda environments.

## Launching the GUI

```bash
conda activate carla_app
python -m gui.main
```

## What the GUI manages

The GUI launches and monitors project commands such as:

* CARLA server
* manual control
* traffic generation
* dataset collection and replay
* OpenPCDet and LaneDet dataset preparation
* OpenPCDet and LaneDet training and evaluation
* benchmarking and prediction replay
* live streaming producer, aggregator, visualizer, and detector nodes

Most processes are started through `conda run -n ... python -m ...`, while the simulator itself is started through `./carla_server.sh`. This keeps the GUI aligned with the same entrypoints documented in the CLI guides.

## Workflow Windows

The home screen opens dedicated workflow windows for:

* `Manual Control` — quick simulator checks, ego spawning, and traffic generation
* `Dataset Collection` — raw run recording and saved-run replay
* `Training` — OpenPCDet and LaneDet dataset preparation, training, and evaluation
* `Benchmark` — benchmark execution and saved prediction replay
* `Live Streaming` — producer, aggregator, visualizer, and detector nodes

Each workflow window includes:

* a workflow-specific session summary
* process status indicators
* per-process logs
* an activity feed
* process start, stop, and restart controls

The main window also provides a `Process Inspector` dialog for cross-workflow process management and a `Clear Logs` action for truncating the GUI-managed log files under `tmp/`.

## Workflow Behavior

The GUI is aware of a few operational dependencies between processes.

For example:

* manual control is always launched with synchronous mode enabled
* traffic generation requires an active manual control process
* dataset collection requires an active manual control process
* the streaming core expects manual control to already be running before the producer is started

The visible `Sync Mode` checkbox in the manual control form is enabled by default and intentionally locked so the user cannot disable it. That keeps the GUI aligned with dataset collection and live streaming requirements.

The live streaming workflow also supports attaching and detaching detector nodes while the producer, aggregator, and visualizer continue running.

## Logs and State

The GUI keeps local runtime state under `tmp/`:

```text
tmp/
├── streaming_gui_logs/
│   ├── server.log
│   ├── manual_control.log
│   └── ...
└── streaming_gui_state.json
```

These files are used as follows:

* `streaming_gui_logs/` — stores one log file per managed process
* `streaming_gui_state.json` — stores the last used arguments, last exit codes, log paths, and the PID of any process that is still alive

When the GUI starts again, it reloads that state and reattaches to processes whose PIDs are still running. This makes it possible to reopen the control center without losing visibility into GUI-managed jobs that were already launched earlier.

## When to Prefer CLI Over GUI

Use the GUI when:

* you want one place to launch and observe a multi-step workflow
* you want logs and activity summaries without juggling multiple terminals
* you are iterating interactively on CARLA, dataset capture, benchmarking, or live inference

Use the CLI directly when:

* you want to copy the documented commands directly and run them step by step
* you are launching training or evaluation jobs on a remote machine
* you want to integrate the workflow into your own terminal-based setup

The documented CLI commands remain the canonical workflow reference. The GUI is a convenience layer built on top of those same entrypoints.
