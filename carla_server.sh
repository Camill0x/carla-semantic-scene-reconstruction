#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CARLA_ROOT:-}" ]]; then
    echo "[error] CARLA_ROOT is not configured"
    echo
    echo "[hint] Add it to your ~/.bashrc, for example:"
    echo "[hint]   export CARLA_ROOT=/path/to/CARLA_0.9.15"
    echo "[hint] Then run: source ~/.bashrc"
    exit 1
fi

if [[ ! -d "${CARLA_ROOT}" ]]; then
    echo "[error] CARLA_ROOT does not point to an existing directory:"
    echo "[error]   ${CARLA_ROOT}"
    exit 1
fi

if [[ ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
    echo "[error] CARLA_ROOT does not look like a CARLA simulator directory:"
    echo "[error]   ${CARLA_ROOT}"
    echo
    echo "[error] Expected executable not found: ${CARLA_ROOT}/CarlaUE4.sh"
    exit 1
fi

cd "${CARLA_ROOT}"

./CarlaUE4.sh "$@"
