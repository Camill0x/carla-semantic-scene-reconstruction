#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/envs/carla_app.yml"
ENV_NAME="carla_app"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[info] repo root: ${ROOT_DIR}"
echo "[info] env file: ${ENV_FILE}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "[1/1] Updating existing conda environment '${ENV_NAME}' from ${ENV_FILE}"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
    echo "[1/1] Creating conda environment '${ENV_NAME}' from ${ENV_FILE}"
    conda env create -f "${ENV_FILE}"
fi

echo "[done] CARLA app environment is ready"
echo "[info] Environment: ${ENV_NAME}"
