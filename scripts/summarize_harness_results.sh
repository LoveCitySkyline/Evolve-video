#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/harness_local_fake}"
PYTHONPATH=src python -m evovideo_skill.cli summarize-results --output-dir "$OUTPUT_DIR"
