#!/bin/bash
set -euo pipefail

CONFIG="${1:-configs/softbomb_config.json}"
DRY_RUN="${DRY_RUN:-0}"

cd "$(dirname "$0")/.."

if [[ "${DRY_RUN}" == "1" ]]; then
  python3 scripts/download_alien.py --config "${CONFIG}" --dry-run
else
  python3 scripts/download_alien.py --config "${CONFIG}"
fi
