#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
CARLA_VERSION="0.9.15"
CARLA_ARCHIVE="${EXTERNAL_DIR}/CARLA_${CARLA_VERSION}.tar.gz"
CARLA_DIR="${EXTERNAL_DIR}/CARLA_${CARLA_VERSION}"
CONDA_ENV_NAME="carla"

SKIP_EXTRACT=0

if [[ "${1:-}" == "--skip-extract" ]]; then
    SKIP_EXTRACT=1
fi

mkdir -p "${EXTERNAL_DIR}"

if [[ "${SKIP_EXTRACT}" -eq 0 ]]; then
  echo "[1/4] Checking CARLA archive"
  if [ ! -f "${CARLA_ARCHIVE}" ]; then
      echo "Archive not found:"
      echo "  ${CARLA_ARCHIVE}"
      echo
      echo "Download CARLA_0.9.15.tar.gz manually and place it in 'external/' directory"
      exit 1
  fi

  if [ ! -f "${CARLA_DIR}/CarlaUE4.sh" ]; then
      echo "[2/4] Extracting CARLA archive into ${CARLA_DIR}"
      mkdir -p "${CARLA_DIR}"
      tar -xzf "${CARLA_ARCHIVE}" -C "${CARLA_DIR}"
  else
      echo "[2/4] CARLA already extracted"
  fi
else
    echo "[1/4] Skipping CARLA archive check"
    echo "[2/4] Skipping CARLA extraction"

    if [ ! -f "${CARLA_DIR}/CarlaUE4.sh" ]; then
        echo "CARLA directory does not look valid:"
        echo "  ${CARLA_DIR}"
        echo
        echo "Expected file not found: ${CARLA_DIR}/CarlaUE4.sh"
        exit 1
    fi
fi

echo "[3/4] Ensuring conda environment exists: ${CONDA_ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "Conda environment '${CONDA_ENV_NAME}' already exists"
    exit 1
else
    conda env create -f "${ROOT_DIR}/envs/carla.yml"
fi

echo "[4/4] Activating conda environment and installing CARLA Python API"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${CARLA_DIR}/PythonAPI/carla/dist"

WHEEL_FILE="$(find . -maxdepth 1 -type f -name 'carla-0.9.15-cp37-*.whl' | head -n 1)"

if [ -z "${WHEEL_FILE}" ]; then
    echo "Could not find CARLA wheel for Python 3.7 in:"
    echo "  ${CARLA_DIR}/PythonAPI/carla/dist"
    exit 1
fi

pip install "${WHEEL_FILE}"

echo
echo "Done."
echo "CARLA directory: ${CARLA_DIR}"
echo "Conda environment: ${CONDA_ENV_NAME}"
