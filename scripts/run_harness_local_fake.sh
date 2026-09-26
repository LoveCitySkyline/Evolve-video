#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/video_skill_graph.toml}"

PYTHONPATH=src python -m evovideo_skill.cli run-harness \
  --config "${CONFIG_PATH}"
