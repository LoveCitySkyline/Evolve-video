#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export EVOVIDEO_PREPARED_CAPABILITIES="configs/prepared_tool_capabilities_style_transfer.json"
export EVOVIDEO_PREPARED_TOOL_STORE="${EVOVIDEO_PREPARED_TOOL_STORE:-outputs/prepared_tool_store_mini15}"
export OPEN_WORLD_STYLE_REPOSITORIES="${OPEN_WORLD_STYLE_REPOSITORIES:-williamyang1991/Rerender_A_Video,omerbt/TokenFlow,RehgLab/RAVE}"
export OPEN_WORLD_ARENA_REPO_LIMIT="${OPEN_WORLD_ARENA_REPO_LIMIT:-2}"
export OPEN_WORLD_MAX_CANDIDATES="${OPEN_WORLD_MAX_CANDIDATES:-3}"
export OPEN_WORLD_MAX_REPOSITORIES_PER_CAPABILITY="${OPEN_WORLD_MAX_REPOSITORIES_PER_CAPABILITY:-8}"

echo "Style-transfer repository continuation"
echo "  store=$EVOVIDEO_PREPARED_TOOL_STORE"
echo "  repositories=$OPEN_WORLD_STYLE_REPOSITORIES"
echo "  arena_limit=$OPEN_WORLD_ARENA_REPO_LIMIT"

exec bash scripts/prepare_complex_video_bench_mini15_tools.sh
