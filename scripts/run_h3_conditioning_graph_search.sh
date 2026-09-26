#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python}" -m evovideo_skill.conditioning_runner \
  --config configs/h3_conditioning_graph_search.json "$@"
