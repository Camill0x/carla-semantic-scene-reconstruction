# Installation

This document describes the installation layout for the project.
CARLA 0.9.15 and the `carla_app` environment provide the runtime layer used for the CARLA Python API, the GUI, dataset tools, and visualization utilities.
The `openpcdet` and `lanedet` environments are only needed when you want to prepare datasets for those model stacks, train or evaluate them, or run the corresponding live detector nodes.

If you are looking for day-to-day commands after setup, continue with:

* [CARLA](CARLA.md) — simulator control, driving, and dataset workflows
* [Training](TRAIN.md) — OpenPCDet and LaneDet preparation, training, and evaluation
* [Benchmarking](BENCHMARK.md) — offline speed tests and prediction replay
* [Streaming](STREAM.md) — live multimodal inference pipeline
* [GUI](GUI.md) — workflow control center for entire project

## Tested Platform

The project has been tested and run on:

* Ubuntu 22.04
* NVIDIA GeForce RTX 2080 Ti (`sm_75`)

This is the reference platform used to validate the setup and model workflows in this repository. CUDA, Python, and package requirements are documented separately for each environment below.

## Repository Setup

Clone the repository and initialize submodules before installing any environments:

```bash
git clone https://github.com/Camill0x/carla-semantic-scene-reconstruction.git
cd carla-semantic-scene-reconstruction

git submodule update --init --recursive
```

## Contents

* [CARLA 0.9.15](#carla-0915)
* [carla_app Environment](#carla_app-environment)
* [OpenPCDet](#openpcdet)
* [LaneDet](#lanedet)

## CARLA 0.9.15

If you do not already have CARLA installed, start here.

Download `CARLA_0.9.15.tar.gz` from the official [CARLA releases page](https://github.com/carla-simulator/carla/releases).

After extracting the archive, choose a permanent location for the simulator directory. The `CARLA_ROOT` variable must point to the extracted folder that contains `CarlaUE4.sh`.

Recommended shell setup:

```bash
echo 'export CARLA_ROOT=/path/to/CARLA_0.9.15' >> ~/.bashrc
source ~/.bashrc
```

Quick verification:

```bash
echo "$CARLA_ROOT"
ls "$CARLA_ROOT/CarlaUE4.sh"
```

If the second command prints the path to `CarlaUE4.sh`, the simulator path is configured correctly.

## carla_app Environment

The `carla_app` environment is the default runtime environment for most operational commands in this repository, including the GUI, dataset tools, Rerun viewers, and CARLA-facing runtime utilities.

### Installation

Run:

```bash
./setup/install_carla_app.sh
```

This script:

* Creates or updates the `carla_app` Conda environment with Python 3.10 from `envs/carla_app.yml`
* Installs the CARLA Python API from PyPI via `carla==0.9.15`
* Installs the runtime and visualization dependencies used by the GUI, dataset tools, and streaming pipeline

Notes:

* If you want Python type stubs for the CARLA API in your editor, take a look at the community-maintained [CARLA Python Stubs](https://github.com/aasewold/carla-python-stubs) project.

## OpenPCDet

The project uses a dedicated Conda environment for OpenPCDet and installs PyTorch separately from the official CUDA 12.1 wheel index.

### Prerequisites

Before installing OpenPCDet, make sure you have:

* Working NVIDIA driver, so `nvidia-smi` succeeds
* CUDA Toolkit `12.1` installed, including `nvcc`
* Exported CUDA paths in your shell
* GPU with compute capability supported by this pinned PyTorch build

For `torch==2.2.2` with `cu121`, the practical upper bound for native targets is `sm_90` (compute capability `9.0`). Before installation, check your GPU on the official NVIDIA list:

* [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)

Recommended CUDA Toolkit download page:

* [CUDA Toolkit 12.1.0 download archive](https://developer.nvidia.com/cuda-12-1-0-download-archive)

Typical shell exports:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

### Installation

Run:

```bash
./setup/install_openpcdet.sh
```

The installer:

* Creates or updates the `openpcdet` Conda environment with Python 3.8 from `envs/openpcdet.yml`
* Installs `torch==2.2.2` and `torchvision==0.17.2` from the official PyTorch `cu121` index
* Installs the `pcdet` package from `third_party/OpenPCDet`

## LaneDet

LaneDet is kept as a third-party submodule under `third_party/lanedet` and uses a dedicated Conda environment because it is pinned to an older PyTorch stack and builds a CUDA extension.

### Prerequisites

Before installing LaneDet, make sure you have:

* Working NVIDIA driver, so `nvidia-smi` succeeds
* CUDA Toolkit installed, including `nvcc`
* Exported CUDA paths in your shell
* GPU with compute capability compatible with this LaneDet environment

The current setup targets `sm_75` (compute capability `7.5`). Check your GPU here:

* [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)

Typical shell exports:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

### Installation

Run:

```bash
./setup/install_lanedet.sh
```

The installer:

* Creates or updates the `lanedet` Conda environment with Python 3.8 from `envs/lanedet.yml`
* Installs `pytorch==1.8.0` and `torchvision==0.9.0` through the environment file
* Installs LaneDet from `third_party/lanedet` in editable mode
