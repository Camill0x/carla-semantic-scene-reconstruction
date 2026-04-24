#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CARLA_DIR="${ROOT_DIR}/external/CARLA_0.9.15"

cd "${CARLA_DIR}"

./CarlaUE4.sh "$@"
