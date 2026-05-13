# CARLA Semantic Scene Reconstruction

CARLA workflow for synchronized dataset collection, detector training, offline benchmarking, and live scene reconstruction with GUI support.

## Overview

* [Introduction](#introduction)
* [Documentation Map](#documentation-map)
* [Environments](#environments)
* [Project Structure](#project-structure)
* [GUI](#gui)
* [Reference Platform](#reference-platform)
* [Results](#results)
* [Third-Party Components](#third-party-components)
* [License](#license)

## Introduction

This repository combines four closely connected workflows:

* CARLA driving, traffic generation, and synchronized raw dataset collection
* Dataset preparation and model training for OpenPCDet and LaneDet
* Offline benchmarking with saved prediction outputs
* Live streaming inference with a lightweight control layer and Rerun visualization

### CARLA Runtime and Dataset Collection

Use CARLA 0.9.15 together with the `carla_app` environment to:

* Launch the simulator
* Drive the ego vehicle in synchronous mode
* Generate surrounding traffic
* Record synchronized RGB, LiDAR, object, lane, and state data
* Replay saved runs in Rerun

### Detector Training and Evaluation

The repository supports two detector stacks:

* `OpenPCDet` for 3D object detection on the project-specific `carla_nuscenes6` dataset family
* `LaneDet` for lane detection on the prepared TuSimple-style dataset built from CARLA runs

Both stacks are integrated as bundled submodules under `third_party/` and are wrapped by project commands for dataset preparation, training, and evaluation.

### Offline Benchmarking

The benchmark workflow measures per-frame model-forward time and end-to-end runtime on recorded CARLA runs. Saved benchmark predictions can be replayed later in Rerun for qualitative inspection.

### Live Streaming Inference

The live pipeline is split into cooperating processes:

* A producer that reads synchronized CARLA sensor data
* Detector nodes that attach to the shared frame stream
* An aggregator that merges predictions into one scene stream
* A Rerun visualizer for live inspection

This keeps the CARLA-facing runtime lightweight while allowing OpenPCDet and LaneDet to stay in their own Conda environments.

## Documentation Map

Start with [Installation](docs/INSTALL.md), then follow the workflow-specific guides below:

* [CARLA](docs/CARLA.md) — simulator control, manual driving, traffic generation, dataset collection, and dataset replay
* [Streaming](docs/STREAM.md) — live producer, aggregator, visualizer, and detector nodes
* [Training](docs/TRAIN.md) — OpenPCDet and LaneDet dataset preparation, training, evaluation, and result layouts
* [Benchmarking](docs/BENCHMARK.md) — benchmark execution and saved prediction replay
* [GUI](docs/GUI.md) — Project Control Center, workflow windows, logs, and process state

## Environments

The project uses three Conda environments:

* `carla_app` — CARLA-facing tools, GUI, dataset utilities, Rerun viewers, and lightweight runtime tooling
* `openpcdet` — OpenPCDet dataset preparation, training, evaluation, benchmarking, and live inference
* `lanedet` — LaneDet dataset preparation, training, evaluation, benchmarking, and live inference

Environment setup and CARLA installation are documented in [Installation](docs/INSTALL.md).

## Project Structure

```text
carla-semantic-scene-reconstruction/
├── carla_server.sh   CARLA server launcher
├── config/           runtime configuration
├── docs/             workflow documentation
├── envs/             Conda environment definitions
├── gui/              Project Control Center
├── setup/            environment installer scripts
├── src/              shared project code
├── tools/            user-facing Python commands
└── third_party/      bundled third-party frameworks
```

## GUI

The repository includes a Project Control Center for interactive workflow control. It launches the same documented CLI commands used throughout the project, but groups them into dedicated workflow windows with process status, logs, and local state handling.

Use the `carla_app` environment to launch it:

```bash
conda activate carla_app
python -m gui.main
```

See [GUI](docs/GUI.md) for details.

## Reference Platform

The project has been tested on:

* operating system: Ubuntu 22.04
* CPU: AMD Ryzen 5 3600X 6-Core Processor
* GPU: NVIDIA GeForce RTX 2080 Ti
* RAM: 32 GB

## Results

### OpenPCDet

| Model | Dataset | mAP | Car AP | Truck AP | Bus AP | Motorcycle AP | Bicycle AP | Pedestrian AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TransFusion-L | `carla_nuscenes6` | `X` | `X` | `X` | `X` | `X` | `X` | `X` |
| CenterPoint-PointPillar | `carla_nuscenes6` | `X` | `X` | `X` | `X` | `X` | `X` | `X` |

### LaneDet

| Model | Dataset | Accuracy | FP | FN | Matched Lane MAE (px) | Matched Lane RMSE (px) | Point MAE (px) | Point RMSE (px) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LaneATT-ResNet34 | `tusimple` | `X` | `X` | `X` | `X` | `X` | `X` | `X` |

### Benchmark Summary

| Model | Model FPS | Runtime FPS | Notes |
| --- | --- | --- | --- |
| OpenPCDet TransFusion-L | `X` | `X` | `X` |
| OpenPCDet CenterPoint-PointPillar | `X` | `X` | `X` |
| LaneDet LaneATT-ResNet34 | `X` | `X` | `X` |

## Third-Party Components

The project includes external frameworks under `third_party/`, including:

* `third_party/OpenPCDet`
* `third_party/lanedet`

Project-specific integrations, dataset adapters, configs, and wrappers are documented in [Training](docs/TRAIN.md).

## License

See the repository root and the bundled third-party components for licensing details.
