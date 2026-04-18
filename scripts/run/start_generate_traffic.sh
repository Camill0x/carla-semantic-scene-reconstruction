#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate carla

python "${ROOT_DIR}/adapters/carla/generate_traffic.py" "$@"
