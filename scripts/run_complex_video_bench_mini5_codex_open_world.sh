#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f benchmarks/complex_video_bench_1k/complex_video_bench_mini5.json ]]; then
  PYTHONPATH=src python scripts/generate_complex_video_bench_mini50.py --preset mini5
fi

export EVOVIDEO_HARNESS_CONFIG="configs/open_world_local_wan_mini5_harness.json"
export VIDEO_OUTPUT_DIR="${EVOVIDEO_MINI5_VIDEO_OUTPUT_DIR:-outputs/harness_open_world_local_wan_mini5_videos}"
export OPEN_WORLD_ARENA_REPO_LIMIT="${EVOVIDEO_MINI5_REPO_LIMIT:-1}"
export OPEN_WORLD_ARENA_MAX_GRAPH_VARIANTS="${EVOVIDEO_MINI5_GRAPH_VARIANTS:-1}"
export OPEN_WORLD_MAX_CANDIDATES="${EVOVIDEO_MINI5_DISCOVERY_CANDIDATES:-1}"
export OPEN_WORLD_MAX_REPOSITORIES_PER_CAPABILITY="${EVOVIDEO_MINI5_REPOSITORIES_PER_CAPABILITY:-1}"
export OPEN_WORLD_GRAPH_IDEAS="${EVOVIDEO_MINI5_GRAPH_IDEAS:-1}"
export OPEN_WORLD_GRAPH_REALIZATION_BUDGET="${EVOVIDEO_MINI5_REALIZATION_BUDGET:-1}"
export EVOVIDEO_MAX_MUTATION_SEARCHES="${EVOVIDEO_MINI5_MUTATION_SEARCHES:-2}"
# max_attempts includes the first deployment, so 3 means two repair attempts.
export CODEX_TOOL_MAX_ATTEMPTS="${EVOVIDEO_MINI5_CODEX_ATTEMPTS:-3}"
export CODEX_TOOL_TIMEOUT_SECONDS="${EVOVIDEO_MINI5_CODEX_TIMEOUT_SECONDS:-300}"
export OPEN_WORLD_VENV_BUILD_TIMEOUT_SECONDS="${EVOVIDEO_MINI5_BUILD_TIMEOUT_SECONDS:-480}"
export OPEN_WORLD_VENV_TOTAL_TIMEOUT_SECONDS="${EVOVIDEO_MINI5_TOTAL_BUILD_TIMEOUT_SECONDS:-600}"
export OPEN_WORLD_VENV_IDLE_TIMEOUT_SECONDS="${EVOVIDEO_MINI5_BUILD_IDLE_TIMEOUT_SECONDS:-90}"
export OPEN_WORLD_VENV_SMOKE_TIMEOUT_SECONDS="${EVOVIDEO_MINI5_SMOKE_TIMEOUT_SECONDS:-480}"
echo "mini5 isolated video output: $VIDEO_OUTPUT_DIR"
echo "mini5 acquisition budget: repos/capability=$OPEN_WORLD_MAX_REPOSITORIES_PER_CAPABILITY codex_attempts/repo=$CODEX_TOOL_MAX_ATTEMPTS"
echo "mini5 mutation validation budget: $EVOVIDEO_MAX_MUTATION_SEARCHES graphs total"
exec bash scripts/run_harness_codex_open_world_venv.sh "$@"
