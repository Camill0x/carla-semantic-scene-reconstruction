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

echo "[info] repo root: ${ROOT_DIR}"
echo "[info] env file: ${ENV_FILE}"
echo "[info] external dir: ${EXTERNAL_DIR}"

if [[ "${SKIP_EXTRACT}" -eq 0 ]]; then
    echo "[1/4] Checking CARLA archive"
    if [[ ! -f "${CARLA_ARCHIVE}" ]]; then
        echo "[error] Archive not found:"
        echo "[error]   ${CARLA_ARCHIVE}"
        echo
        echo "[hint] Download CARLA_0.9.15.tar.gz manually and place it in 'external/' directory"
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
        echo "[error] CARLA directory does not look valid:"
        echo "[error]   ${CARLA_DIR}"
        echo
        echo "[error] Expected file not found: ${CARLA_DIR}/CarlaUE4.sh"
        exit 1
    fi
fi

echo "[info] CARLA dir: ${CARLA_DIR}"

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
    echo "[error] Could not find CARLA wheel for Python 3.7 in:"
    echo "[error]   ${CARLA_DIR}/PythonAPI/carla/dist"
    exit 1
fi

echo "[info] wheel: ${WHEEL_FILE}"
python -m pip install "${WHEEL_FILE}"

echo
echo "[done] CARLA environment is ready"
echo "[info] CARLA directory: ${CARLA_DIR}"
echo "[info] Conda environment: ${CONDA_ENV_NAME}"
