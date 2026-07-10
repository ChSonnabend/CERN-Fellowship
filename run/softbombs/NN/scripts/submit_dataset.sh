#!/bin/bash
set -euo pipefail
CONFIG="${1:-configs/softbomb_config.json}"
python3 scripts/submit.py dataset --config "${CONFIG}"

