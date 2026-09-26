#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRESET="${1:?Usage: $0 mini50|mini100 [run-harness args...]}"
shift

: "${WAN_REPO:?Set WAN_REPO to the local Wan repository containing generate.py}"
: "${WAN_CKPT_DIR:?Set WAN_CKPT_DIR to the local Wan T2V checkpoint directory}"
: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY for Qwen-VL video verification}"

case "$PRESET" in
  mini50)
    CONFIG="configs/frozen_tools_local_wan_mini50_harness.json"
    DEFAULT_WARM_START="outputs/harness_frozen_tools_local_wan_mini15/complex_video_bench_mini15_full"
    DEFAULT_OUTPUT="outputs/harness_frozen_tools_local_wan_mini50"
    DEFAULT_ASSETS="benchmarks/complex_video_bench_1k/assets/mini50"
    DEFAULT_ITERATIONS=12
    DEFAULT_SEARCHES=24
    ;;
  mini100)
    CONFIG="configs/frozen_tools_local_wan_mini100_harness.json"
    DEFAULT_WARM_START="outputs/harness_frozen_tools_local_wan_mini50/complex_video_bench_mini50_full"
    DEFAULT_OUTPUT="outputs/harness_frozen_tools_local_wan_mini100"
    DEFAULT_ASSETS="benchmarks/complex_video_bench_1k/assets/mini100"
    DEFAULT_ITERATIONS=16
    DEFAULT_SEARCHES=32
    ;;
  *)
    echo "Unsupported preset: $PRESET (expected mini50 or mini100)" >&2
    exit 2
    ;;
esac

export GRAPH_PLANNER_BACKEND=codex
export ENABLE_LLM_MUTATION=1
export ENABLE_OPEN_WORLD_TOOLS=0
export RESTORE_CATALOG_TOOLS=1
export ENABLE_MCP_TOOLS=0
export OPEN_WORLD_TOOL_ARENA=1
export OPEN_WORLD_ARENA_MAX_GRAPH_VARIANTS="${OPEN_WORLD_ARENA_MAX_GRAPH_VARIANTS:-2}"
export OPEN_WORLD_ARENA_REQUIRE_PAIR="${OPEN_WORLD_ARENA_REQUIRE_PAIR:-0}"
export EVOVIDEO_MAX_TASK_METRIC_REGRESSION="${EVOVIDEO_MAX_TASK_METRIC_REGRESSION:-0.05}"
export CODEX_BIN="${CODEX_BIN:-codex}"
export CODEX_TOOL_MODEL="${CODEX_TOOL_MODEL:-gpt-5.6-sol}"
export CODEX_TOOL_REASONING_EFFORT="${CODEX_TOOL_REASONING_EFFORT:-high}"
export CODEX_GRAPH_TIMEOUT_SECONDS="${CODEX_GRAPH_TIMEOUT_SECONDS:-180}"
export CODEX_TOOL_HEARTBEAT_SECONDS="${CODEX_TOOL_HEARTBEAT_SECONDS:-30}"
export TOOL_CATALOG_PATH="${EVOVIDEO_PREPARED_TOOL_STORE:-outputs/prepared_tool_store_mini15}/tool_catalog.json"
export VIDEO_OUTPUT_DIR="${EVOVIDEO_SHARED_VIDEO_OUTPUT_DIR:-outputs/harness_frozen_tools_local_wan_mini15_videos}"
export WAN_TIMEOUT_SECONDS="${WAN_TIMEOUT_SECONDS:-900}"
export WAN_OUTPUT_FINALIZE_GRACE_SECONDS="${WAN_OUTPUT_FINALIZE_GRACE_SECONDS:-5}"
export EVOVIDEO_WARM_START_RUN_DIR="${EVOVIDEO_WARM_START_RUN_DIR:-$DEFAULT_WARM_START}"
export EVOVIDEO_AUTO_BOOTSTRAP_ASSETS="${EVOVIDEO_AUTO_BOOTSTRAP_ASSETS:-1}"
export EVOVIDEO_ASSET_BOOTSTRAP_DIR="${EVOVIDEO_ASSET_BOOTSTRAP_DIR:-$DEFAULT_ASSETS}"
export EVOVIDEO_ASSET_BOOTSTRAP_SEED="${EVOVIDEO_ASSET_BOOTSTRAP_SEED:-20260827}"
export EVOVIDEO_ASSET_BOOTSTRAP_FORCE="${EVOVIDEO_ASSET_BOOTSTRAP_FORCE:-0}"
export EVOVIDEO_MAX_MUTATION_SEARCHES="${EVOVIDEO_MAX_MUTATION_SEARCHES:-$DEFAULT_SEARCHES}"
export EVOVIDEO_EVOLUTION_ITERATIONS="${EVOVIDEO_EVOLUTION_ITERATIONS:-$DEFAULT_ITERATIONS}"
export EVOVIDEO_MAX_CANDIDATES="${EVOVIDEO_MAX_CANDIDATES:-2}"
export EVOVIDEO_NO_IMPROVEMENT_LIMIT="${EVOVIDEO_NO_IMPROVEMENT_LIMIT:-12}"
export OPEN_WORLD_GRAPH_IDEAS="${OPEN_WORLD_GRAPH_IDEAS:-4}"
export OPEN_WORLD_GRAPH_REALIZATION_BUDGET="${OPEN_WORLD_GRAPH_REALIZATION_BUDGET:-2}"
export OPEN_WORLD_GRAPH_EXPLORATION_SLOTS="${OPEN_WORLD_GRAPH_EXPLORATION_SLOTS:-2}"
export OPEN_WORLD_GRAPH_MAX_NODES="${OPEN_WORLD_GRAPH_MAX_NODES:-8}"
export GRAPH_LLM_MAX_EDITS="${GRAPH_LLM_MAX_EDITS:-12}"
export VLM_TIMEOUT_SECONDS="${VLM_TIMEOUT_SECONDS:-45}"
export VLM_MAX_IMAGES="${VLM_MAX_IMAGES:-6}"
export EVOVIDEO_VLM_MAX_RETRIES="${EVOVIDEO_VLM_MAX_RETRIES:-1}"

if [[ ! -f "$TOOL_CATALOG_PATH" ]]; then
  echo "Prepared tool catalog is missing: $TOOL_CATALOG_PATH" >&2
  echo "Reuse or sync the mini15 prepared store before scaled evolution." >&2
  exit 1
fi
if [[ ! -d "$EVOVIDEO_WARM_START_RUN_DIR/evolution/registry" ]]; then
  echo "Required warm-start run is missing: $EVOVIDEO_WARM_START_RUN_DIR" >&2
  exit 1
fi
if ! command -v "$CODEX_BIN" >/dev/null 2>&1 && [[ ! -x "$CODEX_BIN" ]]; then
  echo "Codex CLI is unavailable: $CODEX_BIN" >&2
  exit 1
fi
if ! "$CODEX_BIN" login status >/dev/null 2>&1; then
  echo "Codex CLI is not authenticated. Run: $CODEX_BIN login --device-auth" >&2
  exit 1
fi

echo "Scaled frozen-tool graph evolution"
echo "  preset=$PRESET config=$CONFIG output=$DEFAULT_OUTPUT"
echo "  warm_start=$EVOVIDEO_WARM_START_RUN_DIR"
echo "  catalog=$TOOL_CATALOG_PATH"
echo "  shared_video_cache=$VIDEO_OUTPUT_DIR"
echo "  iterations=$EVOVIDEO_EVOLUTION_ITERATIONS mutation_searches=$EVOVIDEO_MAX_MUTATION_SEARCHES"
echo "  candidates/iteration=$EVOVIDEO_MAX_CANDIDATES repository_acquisition=disabled"
echo "  Wan timeout=${WAN_TIMEOUT_SECONDS}s output_finalize_grace=${WAN_OUTPUT_FINALIZE_GRACE_SECONDS}s"

PYTHONPATH=src python -m evovideo_skill.cli tool-preflight --config "$CONFIG"
PYTHONPATH=src python -m evovideo_skill.cli run-harness --config "$CONFIG" "$@"
