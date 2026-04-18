#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CARLA_DIR="${ROOT_DIR}/external/CARLA_0.9.15"

cd "${CARLA_DIR}"

if [ "$#" -eq 0 ]; then
    ./CarlaUE4.sh -quality-level=Low -RenderOffScreen
else
    ./CarlaUE4.sh "$@"
fi
