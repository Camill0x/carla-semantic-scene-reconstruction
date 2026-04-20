#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
CARLA_VERSION="0.9.15"
CARLA_ARCHIVE="${EXTERNAL_DIR}/CARLA_${CARLA_VERSION}.tar.gz"
CARLA_DIR="${EXTERNAL_DIR}/CARLA_${CARLA_VERSION}"
CONDA_ENV_NAME="carla"
ENV_FILE="${ROOT_DIR}/envs/carla.yml"

SKIP_EXTRACT=0

if [[ "${1:-}" == "--skip-extract" ]]; then
    SKIP_EXTRACT=1
fi

if [[ "${SKIP_EXTRACT}" -eq 0 ]]; then
    echo "[1/4] Checking CARLA archive"
    if [[ ! -f "${CARLA_ARCHIVE}" ]]; then
        echo "Archive not found:"
        echo "  ${CARLA_ARCHIVE}"
        echo
        echo "Download CARLA_0.9.15.tar.gz manually and place it in 'external/' directory"
        exit 1
    fi

    if [[ ! -f "${CARLA_DIR}/CarlaUE4.sh" ]]; then
        echo "[2/4] Extracting CARLA archive into ${CARLA_DIR}"
        mkdir -p "${CARLA_DIR}"
        tar -xzf "${CARLA_ARCHIVE}" -C "${CARLA_DIR}"
    else
        echo "[2/4] CARLA already extracted"
    fi
else
    echo "[1/4] Skipping CARLA archive check"
    echo "[2/4] Skipping CARLA extraction"

    if [[ ! -f "${CARLA_DIR}/CarlaUE4.sh" ]]; then
        echo "CARLA directory does not look valid:"
        echo "  ${CARLA_DIR}"
        echo
        echo "Expected file not found: ${CARLA_DIR}/CarlaUE4.sh"
        exit 1
    fi
fi

echo "[3/4] Creating or updating conda environment: ${CONDA_ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
    conda env update -n "${CONDA_ENV_NAME}" -f "${ENV_FILE}" --prune
else
    conda env create -f "${ENV_FILE}"
fi

echo "[4/4] Installing CARLA Python API"
conda activate "${CONDA_ENV_NAME}"

cd "${CARLA_DIR}/PythonAPI/carla/dist"

WHEEL_FILE="$(find . -maxdepth 1 -type f -name 'carla-0.9.15-cp37-*.whl' | head -n 1)"

if [[ -z "${WHEEL_FILE}" ]]; then
    echo "Could not find CARLA wheel for Python 3.7 in:"
    echo "  ${CARLA_DIR}/PythonAPI/carla/dist"
    exit 1
fi

python -m pip install "${WHEEL_FILE}"

echo
echo "Done."
echo "CARLA directory: ${CARLA_DIR}"
echo "Conda environment: ${CONDA_ENV_NAME}"
