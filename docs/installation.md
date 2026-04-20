# Installation

## CARLA 0.9.15

### 1. Download CARLA package

Download `CARLA_0.9.15.tar.gz` from the official [CARLA releases page](https://github.com/carla-simulator/carla/releases) and place it in `external/` directory.

### 2. Install CARLA environment and Python API

Run:

```bash
./scripts/setup/install_carla.sh
```

This script:

* extracts the CARLA package into `external/CARLA_0.9.15/`,
* creates the `carla` conda environment,
* installs required Python dependencies,
* installs the CARLA Python API wheel from `PythonAPI/carla/dist/`.

If you already have CARLA 0.9.15 extracted on your machine, move it to:

```bash
mv /path/to/your/CARLA_0.9.15 ./external/CARLA_0.9.15
```

Then run:

```bash
./scripts/setup/install_carla.sh --skip-extract
```

This skips archive validation and extraction, and only sets up the conda environment and installs the Python API.

## OpenPCDet

The project uses a dedicated Conda environment for OpenPCDet and installs PyTorch separately from the official CUDA 12.1 wheel index.

### Prerequisites

Before installing OpenPCDet, make sure you have:

* a working NVIDIA driver (`nvidia-smi` should succeed),
* CUDA Toolkit `12.1` installed, including `nvcc`,
* exported CUDA paths in your shell,
* a GPU with compute capability supported by this pinned PyTorch build. For `torch==2.2.2` + `cu121`, the practical upper bound for native targets is `sm_90` (compute capability `9.0`). Before installation, check your GPU on the official NVIDIA list: [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus).

Recommended CUDA Toolkit download page:

* [CUDA Toolkit 12.1.0 download archive](https://developer.nvidia.com/cuda-12-1-0-download-archive)


If you already have a working NVIDIA driver on the machine, it is usually better to disable driver installation and install only the CUDA Toolkit.

Typical shell exports:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

### Instalation

Simply run:

```bash
./scripts/setup/install_openpcdet.sh
```

The script:

* creates or updates the `openpcdet` conda environment,
* installs `torch==2.2.2` and `torchvision==0.17.2` from the official PyTorch `cu121` index,
* installs the `pcdet` package from `third_party/OpenPCDet`.
