#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${WAN_REPO:?Set WAN_REPO to the local Wan repository containing generate.py}"
: "${WAN_CKPT_DIR:?Set WAN_CKPT_DIR to the local Wan T2V checkpoint directory}"
: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY for Qwen-VL video verification}"

export GRAPH_PLANNER_BACKEND=codex
export ENABLE_LLM_MUTATION=1
export ENABLE_OPEN_WORLD_TOOLS=0
export RESTORE_CATALOG_TOOLS=1
export ENABLE_MCP_TOOLS=0
export OPEN_WORLD_TOOL_ARENA=1
export OPEN_WORLD_ARENA_MAX_GRAPH_VARIANTS="${OPEN_WORLD_ARENA_MAX_GRAPH_VARIANTS:-2}"
export OPEN_WORLD_ARENA_REQUIRE_PAIR="${OPEN_WORLD_ARENA_REQUIRE_PAIR:-1}"
export EVOVIDEO_MAX_TASK_METRIC_REGRESSION="${EVOVIDEO_MAX_TASK_METRIC_REGRESSION:-0.05}"
export CODEX_BIN="${CODEX_BIN:-codex}"
export CODEX_TOOL_MODEL="${CODEX_TOOL_MODEL:-gpt-5.6-sol}"
export CODEX_TOOL_REASONING_EFFORT="${CODEX_TOOL_REASONING_EFFORT:-high}"
export CODEX_GRAPH_TIMEOUT_SECONDS="${CODEX_GRAPH_TIMEOUT_SECONDS:-180}"
export CODEX_TOOL_HEARTBEAT_SECONDS="${CODEX_TOOL_HEARTBEAT_SECONDS:-30}"
export TOOL_CATALOG_PATH="${EVOVIDEO_PREPARED_TOOL_STORE:-outputs/prepared_tool_store_mini15}/tool_catalog.json"
export VIDEO_OUTPUT_DIR="${EVOVIDEO_MINI15_VIDEO_OUTPUT_DIR:-outputs/harness_frozen_tools_local_wan_mini15_videos}"
export EVOVIDEO_AUTO_BOOTSTRAP_ASSETS="${EVOVIDEO_AUTO_BOOTSTRAP_ASSETS:-1}"
export EVOVIDEO_ASSET_BOOTSTRAP_DIR="${EVOVIDEO_ASSET_BOOTSTRAP_DIR:-benchmarks/complex_video_bench_1k/assets/mini15}"
export EVOVIDEO_ASSET_BOOTSTRAP_SEED="${EVOVIDEO_ASSET_BOOTSTRAP_SEED:-20260827}"
export EVOVIDEO_ASSET_BOOTSTRAP_FORCE="${EVOVIDEO_ASSET_BOOTSTRAP_FORCE:-0}"
export EVOVIDEO_MAX_MUTATION_SEARCHES="${EVOVIDEO_MINI15_MUTATION_SEARCHES:-16}"
export EVOVIDEO_EVOLUTION_ITERATIONS="${EVOVIDEO_MINI15_ITERATIONS:-8}"
export EVOVIDEO_MAX_CANDIDATES="${EVOVIDEO_MINI15_MAX_CANDIDATES:-2}"
export EVOVIDEO_NO_IMPROVEMENT_LIMIT="${EVOVIDEO_MINI15_NO_IMPROVEMENT_LIMIT:-8}"
export OPEN_WORLD_GRAPH_IDEAS="${EVOVIDEO_MINI15_GRAPH_IDEAS:-4}"
export OPEN_WORLD_GRAPH_REALIZATION_BUDGET="${EVOVIDEO_MINI15_REALIZATION_BUDGET:-2}"
export OPEN_WORLD_GRAPH_EXPLORATION_SLOTS="${EVOVIDEO_MINI15_EXPLORATION_SLOTS:-2}"
export OPEN_WORLD_GRAPH_MAX_NODES="${EVOVIDEO_MINI15_GRAPH_MAX_NODES:-8}"
export GRAPH_LLM_MAX_EDITS="${EVOVIDEO_MINI15_GRAPH_MAX_EDITS:-12}"
export VLM_TIMEOUT_SECONDS="${VLM_TIMEOUT_SECONDS:-45}"
export VLM_MAX_IMAGES="${VLM_MAX_IMAGES:-6}"
export EVOVIDEO_VLM_MAX_RETRIES="${EVOVIDEO_VLM_MAX_RETRIES:-1}"

if [[ ! -f "$TOOL_CATALOG_PATH" ]]; then
  echo "Prepared tool catalog is missing: $TOOL_CATALOG_PATH" >&2
  echo "Run: bash scripts/prepare_complex_video_bench_mini15_tools.sh" >&2
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

CONFIG="configs/frozen_tools_local_wan_mini15_harness.json"
echo "Phase B/2: frozen-tool graph evolution"
echo "  catalog=$TOOL_CATALOG_PATH"
echo "  repository acquisition=disabled"
echo "  paired tool arena=enabled (max variants=$OPEN_WORLD_ARENA_MAX_GRAPH_VARIANTS)"
echo "  task-metric regression guard=$EVOVIDEO_MAX_TASK_METRIC_REGRESSION"
echo "  iterations=$EVOVIDEO_EVOLUTION_ITERATIONS candidates/iteration=$EVOVIDEO_MAX_CANDIDATES"
echo "  mutation budget=$EVOVIDEO_MAX_MUTATION_SEARCHES no-improvement-limit=$EVOVIDEO_NO_IMPROVEMENT_LIMIT"
PYTHONPATH=src python -m evovideo_skill.cli tool-preflight --config "$CONFIG"
PYTHONPATH=src python -m evovideo_skill.cli run-harness --config "$CONFIG" "$@"
