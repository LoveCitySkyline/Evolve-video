#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${GRAPH_LLM_MODEL:=openai/gpt-5.6-sol}"
: "${GRAPH_LLM_REASONING_EFFORT:=medium}"
if [[ -z "${GRAPH_LLM_BASE_URL:-}" ]]; then
  case "$GRAPH_LLM_MODEL" in
    */*) GRAPH_LLM_BASE_URL="https://openrouter.ai/api/v1" ;;
    gpt-*|o1*|o3*|o4*) GRAPH_LLM_BASE_URL="https://api.openai.com/v1" ;;
    *) GRAPH_LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" ;;
  esac
fi
if [[ -z "${GRAPH_LLM_API_KEY:-}" ]]; then
  case "$GRAPH_LLM_BASE_URL" in
    *openrouter.ai*) GRAPH_LLM_API_KEY="${OPENROUTER_API_KEY:-}" ;;
    *api.openai.com*) GRAPH_LLM_API_KEY="${OPENAI_API_KEY:-}" ;;
    *)
      case "$GRAPH_LLM_MODEL" in
        gpt-*|o1*|o3*|o4*) GRAPH_LLM_API_KEY="${OPENAI_API_KEY:-}" ;;
        *) GRAPH_LLM_API_KEY="${DASHSCOPE_API_KEY:-}" ;;
      esac
      ;;
  esac
fi
if [[ -z "$GRAPH_LLM_API_KEY" ]]; then
  echo "Set OPENROUTER_API_KEY or GRAPH_LLM_API_KEY for GPT-5.6 Sol planning." >&2
  exit 1
fi
export GRAPH_LLM_API_KEY
export GRAPH_LLM_MODEL GRAPH_LLM_BASE_URL GRAPH_LLM_REASONING_EFFORT
export ENABLE_LLM_MUTATION=1
export TOOL_ACQUISITION_POLICY="${TOOL_ACQUISITION_POLICY:-local_first}"
export ENABLE_MCP_TOOLS="${ENABLE_MCP_TOOLS:-0}"
export MCP_SERVERS_CONFIG="${MCP_SERVERS_CONFIG:-configs/mcp_servers.json}"
export GRAPH_LLM_TIMEOUT_SECONDS="${GRAPH_LLM_TIMEOUT_SECONDS:-300}"
export OPEN_WORLD_SYNTHESIS_TIMEOUT_SECONDS="${OPEN_WORLD_SYNTHESIS_TIMEOUT_SECONDS:-300}"
export GRAPH_LLM_MAX_OUTPUT_TOKENS="${GRAPH_LLM_MAX_OUTPUT_TOKENS:-16384}"
export OPEN_WORLD_SYNTHESIS_MAX_OUTPUT_TOKENS="${OPEN_WORLD_SYNTHESIS_MAX_OUTPUT_TOKENS:-6144}"
export OPEN_WORLD_DISCOVERY_TIMEOUT_SECONDS="${OPEN_WORLD_DISCOVERY_TIMEOUT_SECONDS:-45}"

PYTHONPATH=src python -m evovideo_skill.cli run-harness \
  --config configs/llm_graph_harness.toml "$@"
