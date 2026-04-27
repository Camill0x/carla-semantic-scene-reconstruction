#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LANEDET_DIR="${ROOT_DIR}/third_party/lanedet"
ENV_FILE="${ROOT_DIR}/envs/lanedet.yml"
ENV_NAME="lanedet"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[info] repo root: ${ROOT_DIR}"
echo "[info] env file: ${ENV_FILE}"
echo "[info] LaneDet dir: ${LANEDET_DIR}"

if [[ ! -f "${LANEDET_DIR}/setup.py" ]]; then
    echo "[error] LaneDet submodule does not look initialized:"
    echo "[error]   ${LANEDET_DIR}"
    echo
    echo "[hint] Run: git submodule update --init --recursive third_party/lanedet"
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "[1/3] Updating conda environment '${ENV_NAME}' from ${ENV_FILE}"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
    echo "[1/3] Creating conda environment '${ENV_NAME}' from ${ENV_FILE}"
    conda env create -f "${ENV_FILE}"
fi

echo "[2/3] Activating environment"
conda activate "${ENV_NAME}"

if command -v nvcc >/dev/null 2>&1; then
    CUDA_NVCC="$(command -v nvcc)"
    CUDA_HOME_DEFAULT="$(cd "$(dirname "${CUDA_NVCC}")/.." && pwd)"
    export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_DEFAULT}}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
    echo "[info] CUDA_HOME=${CUDA_HOME}"
    echo "[info] nvcc: ${CUDA_NVCC}"
else
    echo "[warn] nvcc not found; LaneDet CUDA extensions may fail to build"
fi

echo "[3/3] Installing the LaneDet Python package from third_party/lanedet"
cd "${LANEDET_DIR}"
python -m pip install --no-deps -e .

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

echo "[done] LaneDet environment is ready"
echo "[info] Environment: ${ENV_NAME}"
echo "[info] Repo: ${LANEDET_DIR}"
