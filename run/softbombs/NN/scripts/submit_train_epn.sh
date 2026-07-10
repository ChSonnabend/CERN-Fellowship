#!/bin/bash
set -euo pipefail
CONFIG="${1:-configs/epn_config.json}"
python3 scripts/submit.py train --config "${CONFIG}"
