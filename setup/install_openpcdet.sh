#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPCDET_DIR="${ROOT_DIR}/third_party/OpenPCDet"
ENV_FILE="${ROOT_DIR}/envs/openpcdet.yml"
ENV_NAME="openpcdet"
TORCH_VERSION="2.2.2"
TORCHVISION_VERSION="0.17.2"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[info] repo root: ${ROOT_DIR}"
echo "[info] env file: ${ENV_FILE}"
echo "[info] OpenPCDet dir: ${OPENPCDET_DIR}"

if [[ ! -f "${OPENPCDET_DIR}/setup.py" ]]; then
    echo "[error] OpenPCDet submodule does not look initialized:"
    echo "[error]   ${OPENPCDET_DIR}"
    echo
    echo "[hint] Run: git submodule update --init --recursive third_party/OpenPCDet"
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[error] nvcc not found in PATH"
    echo "[hint] Install CUDA Toolkit 12.1 and export CUDA_HOME, PATH and LD_LIBRARY_PATH first"
    exit 1
fi

CUDA_NVCC="$(command -v nvcc)"
CUDA_HOME_DEFAULT="$(cd "$(dirname "${CUDA_NVCC}")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_DEFAULT}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "[info] CUDA_HOME=${CUDA_HOME}"
echo "[info] nvcc: ${CUDA_NVCC}"
nvcc --version

if ! nvcc --version | grep -q "release 12.1"; then
    echo "[warn] nvcc does not report CUDA 12.1"
    echo "[warn] This setup is pinned to torch ${TORCH_VERSION} / torchvision ${TORCHVISION_VERSION} from the cu121 index"
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "[1/5] Updating existing conda environment '${ENV_NAME}' from ${ENV_FILE}"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
    echo "[1/5] Creating conda environment '${ENV_NAME}' from ${ENV_FILE}"
    conda env create -f "${ENV_FILE}"
fi

echo "[2/5] Activating environment"
conda activate "${ENV_NAME}"

echo "[3/5] Installing PyTorch wheels from the official cu121 index"
python -m pip install --upgrade pip
python -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"

echo "[4/5] Installing the OpenPCDet Python package from third_party/OpenPCDet"
cd "${OPENPCDET_DIR}"
python -m pip install -e .

echo "[5/5] Checking GPU compute capability against the installed torch build"
python - <<'PY'
import torch

print(f"[info] torch: {torch.__version__}")
print(f"[info] cuda is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    arch = f"sm_{capability[0]}{capability[1]}"
    supported_archs = torch.cuda.get_arch_list()

    print(f"[info] GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"[info] compute capability: {capability[0]}.{capability[1]} ({arch})")
    print(f"[info] torch supported architectures: {supported_archs}")

    if arch in supported_archs:
        print(f"[info] {arch} is supported by the installed torch build")
    else:
        print(f"[warn] {arch} is not listed in torch.cuda.get_arch_list()")
        print("[warn] You may see compatibility warnings or need a newer CUDA/PyTorch stack")
else:
    print("[warn] CUDA device not visible to torch during verification")
PY

echo "[done] OpenPCDet environment is ready"
echo "[info] Environment: ${ENV_NAME}"
echo "[info] Repo: ${OPENPCDET_DIR}"
