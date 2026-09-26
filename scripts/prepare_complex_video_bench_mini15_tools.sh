#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${WAN_REPO:?Set WAN_REPO to the local Wan repository containing generate.py}"
: "${WAN_CKPT_DIR:?Set WAN_CKPT_DIR to the local Wan T2V checkpoint directory}"

export GRAPH_PLANNER_BACKEND=codex
export OPEN_WORLD_ACQUISITION_AGENT=codex
export ENABLE_OPEN_WORLD_TOOLS=1
export RESTORE_CATALOG_TOOLS=0
export ENABLE_MCP_TOOLS=0
export OPEN_WORLD_SANDBOX_BACKEND="${OPEN_WORLD_SANDBOX_BACKEND:-auto}"
export OPEN_WORLD_MICROMAMBA_BIN="${OPEN_WORLD_MICROMAMBA_BIN:-micromamba}"
export OPEN_WORLD_VENV_AUTO_APPROVE="${OPEN_WORLD_VENV_AUTO_APPROVE:-1}"
export OPEN_WORLD_VENV_INTERACTIVE_APPROVAL=0
export OPEN_WORLD_TOOL_ARENA="${OPEN_WORLD_TOOL_ARENA:-1}"
export OPEN_WORLD_ARENA_REPO_LIMIT="${OPEN_WORLD_ARENA_REPO_LIMIT:-2}"
export OPEN_WORLD_MAX_CANDIDATES="${OPEN_WORLD_MAX_CANDIDATES:-2}"
export OPEN_WORLD_MAX_REPOSITORIES_PER_CAPABILITY="${OPEN_WORLD_MAX_REPOSITORIES_PER_CAPABILITY:-6}"
export OPEN_WORLD_STYLE_REPOSITORIES="${OPEN_WORLD_STYLE_REPOSITORIES:-williamyang1991/Rerender_A_Video,omerbt/TokenFlow,RehgLab/RAVE}"
export OPEN_WORLD_GPU_SMOKE_REQUIRED="${OPEN_WORLD_GPU_SMOKE_REQUIRED:-1}"
export OPEN_WORLD_REQUIRE_PINNED_MODEL_REVISION="${OPEN_WORLD_REQUIRE_PINNED_MODEL_REVISION:-1}"
export OPEN_WORLD_REQUIRE_MODEL_LOAD_SMOKE="${OPEN_WORLD_REQUIRE_MODEL_LOAD_SMOKE:-1}"
export OPEN_WORLD_DISCOVERY_TIMEOUT_SECONDS="${OPEN_WORLD_DISCOVERY_TIMEOUT_SECONDS:-15}"
export OPEN_WORLD_VENV_TOTAL_TIMEOUT_SECONDS="${OPEN_WORLD_VENV_TOTAL_TIMEOUT_SECONDS:-600}"
export OPEN_WORLD_VENV_BUILD_TIMEOUT_SECONDS="${OPEN_WORLD_VENV_BUILD_TIMEOUT_SECONDS:-480}"
export OPEN_WORLD_VENV_IDLE_TIMEOUT_SECONDS="${OPEN_WORLD_VENV_IDLE_TIMEOUT_SECONDS:-60}"
export OPEN_WORLD_VENV_SMOKE_TIMEOUT_SECONDS="${OPEN_WORLD_VENV_SMOKE_TIMEOUT_SECONDS:-480}"
export OPEN_WORLD_PIP_TIMEOUT_SECONDS="${OPEN_WORLD_PIP_TIMEOUT_SECONDS:-30}"
export OPEN_WORLD_PIP_RETRIES="${OPEN_WORLD_PIP_RETRIES:-1}"
export CODEX_BIN="${CODEX_BIN:-codex}"
export CODEX_TOOL_MODEL="${CODEX_TOOL_MODEL:-gpt-5.6-sol}"
export CODEX_TOOL_REASONING_EFFORT="${CODEX_TOOL_REASONING_EFFORT:-high}"
export CODEX_TOOL_SEARCH="${CODEX_TOOL_SEARCH:-1}"
export CODEX_TOOL_APPROVAL_MODE="${CODEX_TOOL_APPROVAL_MODE:-auto-review}"
export CODEX_TOOL_TIMEOUT_SECONDS="${CODEX_TOOL_TIMEOUT_SECONDS:-180}"
export CODEX_TOOL_MAX_ATTEMPTS="${CODEX_TOOL_MAX_ATTEMPTS:-2}"

case "$OPEN_WORLD_SANDBOX_BACKEND" in
  venv|auto)
    if [[ -z "${OPEN_WORLD_VENV_PYTHON:-}" ]]; then
      for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import ensurepip, venv' >/dev/null 2>&1; then
          export OPEN_WORLD_VENV_PYTHON="$(command -v "$candidate")"
          break
        fi
      done
    fi
    if [[ -z "${OPEN_WORLD_VENV_PYTHON:-}" && "$OPEN_WORLD_SANDBOX_BACKEND" == "venv" ]]; then
      echo "Venv preparation requires Python with ensurepip and venv." >&2
      exit 1
    fi
    if [[ "$OPEN_WORLD_SANDBOX_BACKEND" == "auto" ]] \
      && ! command -v "$OPEN_WORLD_MICROMAMBA_BIN" >/dev/null 2>&1 \
      && [[ ! -x "$OPEN_WORLD_MICROMAMBA_BIN" ]]; then
      echo "Warning: micromamba is unavailable; legacy Python/Torch repositories may be skipped." >&2
    fi
    ;;
  micromamba)
    if ! command -v "$OPEN_WORLD_MICROMAMBA_BIN" >/dev/null 2>&1 \
      && [[ ! -x "$OPEN_WORLD_MICROMAMBA_BIN" ]]; then
      echo "Micromamba preparation selected but binary is unavailable: $OPEN_WORLD_MICROMAMBA_BIN" >&2
      exit 1
    fi
    ;;
  docker|search-only|search_only)
    ;;
  *)
    echo "Unsupported OPEN_WORLD_SANDBOX_BACKEND=$OPEN_WORLD_SANDBOX_BACKEND" >&2
    exit 1
    ;;
esac

if ! command -v "$CODEX_BIN" >/dev/null 2>&1 && [[ ! -x "$CODEX_BIN" ]]; then
  echo "Codex CLI is unavailable: $CODEX_BIN" >&2
  exit 1
fi
if ! "$CODEX_BIN" login status >/dev/null 2>&1; then
  echo "Codex CLI is not authenticated. Run: $CODEX_BIN login --device-auth" >&2
  exit 1
fi
if [[ -n "${OPEN_WORLD_GIT_CAINFO:-}" ]]; then
  export GIT_SSL_CAINFO="$OPEN_WORLD_GIT_CAINFO"
fi
if [[ "${OPEN_WORLD_GIT_PREFLIGHT:-1}" == "1" ]]; then
  git ls-remote https://github.com/git/git.git HEAD >/dev/null
fi

STORE_DIR="${EVOVIDEO_PREPARED_TOOL_STORE:-outputs/prepared_tool_store_mini15}"
CAPABILITIES="${EVOVIDEO_PREPARED_CAPABILITIES:-configs/prepared_tool_capabilities_mini15.json}"
echo "Phase A/2: prepare and verify repository tools"
echo "  store=$STORE_DIR"
echo "  capabilities=$CAPABILITIES"
echo "  repository variants/capability=$OPEN_WORLD_ARENA_REPO_LIMIT"
echo "  persistent repository attempt budget/capability=$OPEN_WORLD_MAX_REPOSITORIES_PER_CAPABILITY"
echo "  target capability families: i2v, multishot identity, motion control, global/region editing, segment repair, style transfer, deflicker, audio-conditioned generation"
if [[ "${EVOVIDEO_PREPARE_ALLOW_PARTIAL:-0}" == "1" ]]; then
  PYTHONPATH=src python -m evovideo_skill.cli prepare-tools \
    --config configs/open_world_local_wan_mini15_harness.json \
    --capabilities "$CAPABILITIES" \
    --store-dir "$STORE_DIR" \
    --allow-partial
else
  PYTHONPATH=src python -m evovideo_skill.cli prepare-tools \
    --config configs/open_world_local_wan_mini15_harness.json \
    --capabilities "$CAPABILITIES" \
    --store-dir "$STORE_DIR"
fi
