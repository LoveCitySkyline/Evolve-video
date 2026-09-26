#!/usr/bin/env bash
set -euo pipefail
EVOVIDEO_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EVOVIDEO_REPO_ROOT"
export PYTHONPATH="$EVOVIDEO_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PROVIDER=local-h3
exec "${EVOVIDEO_PYTHON:-python}" -m evovideo_skill.h3_cli run \
  --config configs/h3_local_graph_harness.json "$@"
