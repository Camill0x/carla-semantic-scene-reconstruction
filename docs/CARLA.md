# Running

This document covers the operational, non-training part of the project: starting CARLA, driving the ego vehicle, generating traffic, collecting raw datasets, and replaying saved runs.

Before using these commands, complete the setup described in [Installation](INSTALL.md). In practice, you need a valid `CARLA_ROOT` and an installed `carla_app` environment.

All workflows described here are also available through the GUI. If you want to use the graphical workflow layer instead of driving everything from the terminal, continue with [GUI](GUI.md).

## Contents

* [CARLA Server](#carla-server)
* [Manual Control](#manual-control)
* [Traffic Generation](#traffic-generation)
* [Raw Dataset Collection](#raw-dataset-collection)
* [Dataset Viewer](#dataset-viewer)

## CARLA Server

The wrapper launches `CarlaUE4.sh` from the simulator directory pointed to by `CARLA_ROOT`.

### Useful flags

* `-RenderOffScreen` — runs without the on-screen window
* `-ResX=... -ResY=...` — changes the render resolution
* `-quality-level=Low` — lowers visual quality and may reduce GPU load

### Examples

```bash
# Start the simulator with default settings.
./carla_server.sh

# Run without an on-screen window.
./carla_server.sh -RenderOffScreen

# Lower resolution and visual quality for a lighter run.
./carla_server.sh -ResX=800 -ResY=600 -quality-level=Low
```

### Notes

* Avoid using `-RenderOffScreen` and `-quality-level=Low` together. In CARLA 0.9.15 this combination may lead to crashes when changing maps.

## Manual Control

This entrypoint is a project-owned variant of the CARLA manual control client. It uses the CARLA connection configured in: `config/runtime.json`

### Useful flags

* `--sync` — required for dataset collection and for the live producer
* `--map` — map to load before spawning the ego vehicle (default: `Town10HD`)
* `--filter` — select the ego vehicle blueprint (default: `vehicle.*`)
* `--autopilot` — start in autopilot mode (default: `False`)
* `--client-fps` — FPS of the client window (default: `20`)
* `--delta-seconds` — simulation step (default: `0.05` → 20 FPS)
* `--sim-fps` — alternative way to control simulation FPS (overrides `--delta-seconds`)

### Examples

```bash
conda activate carla_app

# Start synchronized manual control with default map and vehicle selection.
python -m tools.carla.manual_control --sync

# Load a specific map and pin the ego vehicle blueprint.
python -m tools.carla.manual_control \
  --sync \
  --map Town10HD \
  --filter vehicle.dodge.charger_2020

# Drive with sync mode, autopilot, and an explicit client/simulation FPS.
python -m tools.carla.manual_control \
  --sync \
  --autopilot \
  --client-fps 20 \
  --sim-fps 20
```

### Basic controls

* `W`, `A`, `S`, `D` — driving
* `Q` — reverse gear
* `P` — toggle autopilot
* `Backspace` — change vehicle
* `C` — change weather
* `ESC` — exit

## Traffic Generation

Spawn background traffic. This entrypoint is based on the original CARLA traffic-generation script and adapted for this project workflow.

Walker collision handling has also been re-enabled here. In the original CARLA script it is disabled by default, which caused pedestrians not to be detected correctly by LiDAR in this project setup.

### Useful flags

* `-n` — number of vehicles
* `-w` — number of walkers

### Examples

```bash
conda activate carla_app

# Start traffic generation with 20 vehicles and 10 walkers
python -m tools.carla.generate_traffic -n 20 -w 10
```

## Raw Dataset Collection

This tool records synchronized multimodal frames from the active CARLA session and saves them into a new raw dataset run.

Each captured frame includes the front RGB camera image, the LiDAR point cloud, GT object annotations, lane annotations, and the metadata needed later by the training, benchmarking, and visualization workflows.

During collection, the tool:

* Waits for synchronized CARLA ticks
* Attaches a LiDAR and front RGB camera to the `hero` vehicle
* Saves points, RGB frames, GT objects, lane annotations, and metadata
* Writes a new `run_XXXX` directory under the configured raw dataset root

### Flags

* `-n`, `--num-frames` — control how many frames are saved
* `--every-nth` — store only every N-th synchronized frame

### Requirements

* CARLA must already be running
* Manual control must already be active in `--sync` mode

### Examples

```bash
conda activate carla_app

# Collect 100 frames while skipping intermediate CARLA ticks.
python -m tools.dataset.collect_dataset --num-frames 100 --every-nth 20

# Capture a longer run with denser sampling.
python -m tools.dataset.collect_dataset --num-frames 500 --every-nth 5
```

### Output layout

A collected raw run looks like this:

```text
datasets/
└── raw/
    └── run_0001/
        ├── frame_000000/
        │   ├── meta.json
        │   ├── points.npy
        │   ├── front_rgb.png
        │   ├── gt_boxes.npy
        │   ├── gt_names.npy
        │   ├── ego_box.npy
        │   ├── objects.json
        │   └── lanes.json
        └── frame_000001/
            └── ...
```

Per frame:

* `meta.json` — stores frame index, CARLA frame number, map, sensor metadata, and runtime context
* `points.npy` — stores LiDAR point cloud
* `front_rgb.png` — stores front camera image
* `gt_boxes.npy` — stores ground truth 3D bounding boxes of the objects
* `gt_names.npy` — stores ground truth class names of the objects
* `objects.json` — stores structured object-level annotations
* `lanes.json` — stores collected lane geometry and related metadata

The exact sensor defaults and runtime settings come from `config/runtime.json`

## Dataset Viewer

This tool loads all `frame_*` directories from one raw run and replays them as a timed offline scene.

### Flags

* `--run-dir` — select saved raw dataset run
* `--fps` — control playback speed (default: `20`)
* `--show-grid` — overlay the 3D ground grid (default: `False`)

### Examples

```bash
conda activate carla_app

# Replay a run with default viewer settings.
python -m tools.dataset.show_dataset --run-dir datasets/raw/run_0001

# Slow the playback down for inspection.
python -m tools.dataset.show_dataset --run-dir datasets/raw/run_0001 --fps 5

# Replay with the 3D ground grid enabled.
python -m tools.dataset.show_dataset \
  --run-dir datasets/raw/run_0001 \
  --fps 20 \
  --show-grid
```
