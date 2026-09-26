#!/usr/bin/env bash
set -euo pipefail

: "${WAN_REPO:?Set WAN_REPO to the local Wan repo path containing generate.py}"
: "${WAN_CKPT_DIR:?Set WAN_CKPT_DIR to the local Wan checkpoint directory}"

PYTHONPATH=src python -m evovideo_skill.cli run-harness \
  --config configs/local_wan_harness.json
