#!/bin/bash
set -euo pipefail
CONFIG="${1:-configs/test_config.json}"
python3 scripts/submit.py test-dataset --config "${CONFIG}"

