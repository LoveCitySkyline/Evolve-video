#!/usr/bin/env bash
set -euo pipefail

export PROVIDER="${PROVIDER:-aliyun-wanx}"

EVOVIDEO_CONFIG="${EVOVIDEO_CONFIG:-configs/api_smoke_harness.json}"
PYTHONPATH=src python -m evovideo_skill.cli run-harness --config "$EVOVIDEO_CONFIG"
