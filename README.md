# CARLA Semantic Scene Reconstruction

Multimodal CARLA workflow for synchronized dataset collection, detector training, offline benchmarking, and live scene reconstruction with GUI-assisted orchestration.

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
* dataset preparation and model training for OpenPCDet and LaneDet
* offline benchmarking with replayable prediction outputs
* live streaming inference with a lightweight control layer and Rerun visualization

### CARLA Runtime and Dataset Collection

Use CARLA 0.9.15 together with the `carla_app` environment to:

* launch the simulator
* drive the ego vehicle in synchronous mode
* generate surrounding traffic
* record synchronized RGB, LiDAR, object, lane, and state data
* replay saved runs in Rerun

### Detector Training and Evaluation

The repository supports two detector stacks:

* `OpenPCDet` for 3D object detection on the project-specific `carla_nuscenes6` dataset family
* `LaneDet` for lane detection on the prepared TuSimple-style dataset built from CARLA runs

Both stacks are integrated as vendored submodules under `third_party/` and are wrapped by project-owned entrypoints for dataset preparation, training, and evaluation.

### Offline Benchmarking

The benchmark workflow measures per-frame model-forward time and end-to-end runtime on recorded CARLA runs. Saved benchmark predictions can be replayed later in Rerun for qualitative inspection.

### Live Streaming Inference

The live pipeline is split into cooperating processes:

* a producer that reads synchronized CARLA sensor data
* detector nodes that attach to the shared frame stream
* an aggregator that merges predictions into one scene stream
* a Rerun visualizer for live inspection

This keeps the CARLA-facing runtime lightweight while allowing OpenPCDet and LaneDet to stay in their own Conda environments.

## Documentation Map

Start with [docs/installation.md](docs/installation.md), then follow the workflow-specific guides below:

* [docs/running.md](docs/running.md) — simulator control, manual driving, traffic generation, dataset collection, and dataset replay
* [docs/streaming.md](docs/streaming.md) — live producer, aggregator, visualizer, and detector nodes
* [docs/training.md](docs/training.md) — OpenPCDet and LaneDet dataset preparation, training, evaluation, and result layouts
* [docs/benchmarking.md](docs/benchmarking.md) — benchmark execution and saved prediction replay
* [docs/gui.md](docs/gui.md) — Project Control Center, workflow windows, logs, and process state

## Environments

The project uses three Conda environments:

* `carla_app` — CARLA-facing tools, GUI, dataset utilities, Rerun viewers, and lightweight runtime tooling
* `openpcdet` — OpenPCDet dataset preparation, training, evaluation, benchmarking, and live inference
* `lanedet` — LaneDet dataset preparation, training, evaluation, benchmarking, and live inference

Environment setup and CARLA installation are documented in [docs/installation.md](docs/installation.md).

## Project Structure

```text
carla-semantic-scene-reconstruction/
├── carla_server.sh   CARLA server launcher
├── config/           runtime configuration
├── docs/             workflow-oriented documentation
├── envs/             Conda environment definitions
├── gui/              Project Control Center
├── setup/            environment installer scripts
├── src/              shared project code
├── tools/            user-facing Python entrypoints
└── third_party/      vendored external frameworks
```

## GUI

The repository includes a Project Control Center for interactive workflow orchestration. It launches the same documented CLI entrypoints used throughout the project, but groups them into dedicated workflow windows with process status, logs, and local state handling.

Use the `carla_app` environment to launch it:

```bash
conda activate carla_app
python -m gui.main
```

See [docs/gui.md](docs/gui.md) for details.

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

The project vendors external frameworks under `third_party/`, including:

* `third_party/OpenPCDet`
* `third_party/lanedet`

Project-specific integrations, dataset adapters, configs, and wrappers are documented in [docs/training.md](docs/training.md).

## License

See the repository root and the bundled third-party components for licensing details.
