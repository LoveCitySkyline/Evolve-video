#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${WAN_REPO:?Set WAN_REPO to the local Wan repository containing generate.py}"
: "${WAN_CKPT_DIR:?Set WAN_CKPT_DIR to the local Wan T2V checkpoint directory}"
: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY for Qwen-VL video verification}"

export ENABLE_LLM_MUTATION=0
export ENABLE_OPEN_WORLD_TOOLS=0
export RESTORE_CATALOG_TOOLS=1
export ENABLE_MCP_TOOLS=0
export TOOL_CATALOG_PATH="${EVOVIDEO_PREPARED_TOOL_STORE:-outputs/prepared_tool_store_mini15}/tool_catalog.json"
export VIDEO_OUTPUT_DIR="${EVOVIDEO_GAIN_SMOKE_VIDEO_OUTPUT_DIR:-outputs/harness_frozen_tools_local_wan_gain_smoke2_videos}"
export EVOVIDEO_AUTO_BOOTSTRAP_ASSETS=1
export EVOVIDEO_ASSET_BOOTSTRAP_DIR="${EVOVIDEO_ASSET_BOOTSTRAP_DIR:-benchmarks/complex_video_bench_1k/assets/mini15}"
export EVOVIDEO_ASSET_BOOTSTRAP_SEED="${EVOVIDEO_ASSET_BOOTSTRAP_SEED:-20260827}"
export EVOVIDEO_ASSET_BOOTSTRAP_FORCE=0
export EVOVIDEO_MAX_MUTATION_SEARCHES=2
export EVOVIDEO_EVOLUTION_ITERATIONS=1
export EVOVIDEO_MAX_CANDIDATES=2
export EVOVIDEO_NO_IMPROVEMENT_LIMIT=1
export EVOVIDEO_COMPARISON_TOP_K="${EVOVIDEO_COMPARISON_TOP_K:-2}"
export VLM_TIMEOUT_SECONDS="${VLM_TIMEOUT_SECONDS:-45}"
export VLM_MAX_IMAGES="${VLM_MAX_IMAGES:-6}"
export EVOVIDEO_VLM_MAX_RETRIES="${EVOVIDEO_VLM_MAX_RETRIES:-1}"

if [[ ! -f "$TOOL_CATALOG_PATH" ]]; then
  echo "Prepared tool catalog is missing: $TOOL_CATALOG_PATH" >&2
  echo "Run Phase 1 first: bash scripts/prepare_complex_video_bench_mini15_tools.sh" >&2
  exit 1
fi

CONFIG="configs/frozen_tools_local_wan_gain_smoke2_harness.json"
echo "Diagnostic gain smoke: source video -> RAVE -> temporal deflicker"
echo "  catalog=$TOOL_CATALOG_PATH"
echo "  repository acquisition=disabled"
echo "  LLM graph planning=disabled"
PYTHONPATH=src python -m evovideo_skill.cli tool-preflight --config "$CONFIG"
PYTHONPATH=src python -m evovideo_skill.cli run-harness --config "$CONFIG" "$@"
