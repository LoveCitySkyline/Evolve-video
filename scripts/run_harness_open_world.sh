#!/usr/bin/env bash
set -euo pipefail

: "${WAN_REPO:?Set WAN_REPO to the local Wan repository containing generate.py}"
: "${WAN_CKPT_DIR:?Set WAN_CKPT_DIR to the local Wan T2V checkpoint directory}"

: "${GRAPH_PLANNER_BACKEND:=api}"
: "${OPEN_WORLD_ACQUISITION_AGENT:=llm}"
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
NEEDS_GRAPH_API=0
[[ "$GRAPH_PLANNER_BACKEND" == "api" ]] && NEEDS_GRAPH_API=1
[[ "$OPEN_WORLD_ACQUISITION_AGENT" == "llm" ]] && NEEDS_GRAPH_API=1
if [[ "$NEEDS_GRAPH_API" == "1" && -z "$GRAPH_LLM_API_KEY" ]]; then
  echo "Set OPENROUTER_API_KEY or GRAPH_LLM_API_KEY for GPT-5.6 Sol planning and adapter synthesis." >&2
  exit 1
fi
export GRAPH_PLANNER_BACKEND OPEN_WORLD_ACQUISITION_AGENT
export GRAPH_LLM_MODEL GRAPH_LLM_BASE_URL GRAPH_LLM_REASONING_EFFORT GRAPH_LLM_API_KEY

if [[ "$GRAPH_PLANNER_BACKEND" == "codex" || "$OPEN_WORLD_ACQUISITION_AGENT" == "codex" ]]; then
  CODEX_BIN="${CODEX_BIN:-codex}"
  if ! command -v "$CODEX_BIN" >/dev/null 2>&1 && [[ ! -x "$CODEX_BIN" ]]; then
    echo "Codex backend selected but CODEX_BIN is unavailable: $CODEX_BIN" >&2
    exit 1
  fi
  if ! "$CODEX_BIN" login status >/dev/null 2>&1; then
    echo "Codex CLI is not authenticated. Run: $CODEX_BIN login --device-auth" >&2
    exit 1
  fi
  export CODEX_BIN
  echo "Codex agent backend: authenticated"
  echo "  graph_planner=$GRAPH_PLANNER_BACKEND"
  echo "  tool_acquisition=$OPEN_WORLD_ACQUISITION_AGENT"
fi

OPEN_WORLD_SANDBOX_BACKEND="${OPEN_WORLD_SANDBOX_BACKEND:-docker}"
export OPEN_WORLD_SANDBOX_BACKEND
case "$OPEN_WORLD_SANDBOX_BACKEND" in
  docker)
    if ! command -v "${OPEN_WORLD_DOCKER_BIN:-docker}" >/dev/null 2>&1; then
      echo "Docker backend selected but Docker is unavailable. Set OPEN_WORLD_SANDBOX_BACKEND=venv or search-only." >&2
      exit 1
    fi
    ;;
  venv|auto)
    VENV_PYTHON_CANDIDATES=()
    if [[ -n "${OPEN_WORLD_VENV_PYTHON:-}" ]]; then
      VENV_PYTHON_CANDIDATES+=("$OPEN_WORLD_VENV_PYTHON")
    else
      command -v python3 >/dev/null 2>&1 && VENV_PYTHON_CANDIDATES+=("$(command -v python3)")
      command -v python >/dev/null 2>&1 && VENV_PYTHON_CANDIDATES+=("$(command -v python)")
    fi
    VENV_PYTHON_FOUND=""
    for candidate in "${VENV_PYTHON_CANDIDATES[@]}"; do
      if "$candidate" -c 'import ensurepip, venv' >/dev/null 2>&1; then
        VENV_PYTHON_FOUND="$candidate"
        break
      fi
    done
    if [[ -z "$VENV_PYTHON_FOUND" ]]; then
      if [[ "$OPEN_WORLD_SANDBOX_BACKEND" == "venv" ]]; then
        echo "Venv backend requires a Python interpreter with ensurepip and venv." >&2
        echo "Install python3-venv or set OPEN_WORLD_VENV_PYTHON=/path/to/a/compatible/python." >&2
        exit 1
      fi
      echo "No host venv Python found; auto backend will require micromamba." >&2
    else
      export OPEN_WORLD_VENV_PYTHON="$VENV_PYTHON_FOUND"
      echo "Open-world host Python: $OPEN_WORLD_VENV_PYTHON"
    fi
    if [[ "$OPEN_WORLD_SANDBOX_BACKEND" == "auto" ]] && ! command -v "${OPEN_WORLD_MICROMAMBA_BIN:-micromamba}" >/dev/null 2>&1; then
      echo "Open-world auto backend: micromamba unavailable; compatible tools will still use host venv." >&2
      echo "  Legacy Python/Torch tools require OPEN_WORLD_MICROMAMBA_BIN or micromamba on PATH." >&2
    fi
    ;;
  micromamba)
    if ! command -v "${OPEN_WORLD_MICROMAMBA_BIN:-micromamba}" >/dev/null 2>&1 && [[ ! -x "${OPEN_WORLD_MICROMAMBA_BIN:-micromamba}" ]]; then
      echo "Micromamba backend selected but OPEN_WORLD_MICROMAMBA_BIN is unavailable." >&2
      exit 1
    fi
    ;;
  search-only|search_only)
    ;;
  *)
    echo "OPEN_WORLD_SANDBOX_BACKEND must be docker, auto, venv, micromamba, or search-only." >&2
    exit 1
    ;;
esac

export ENABLE_LLM_MUTATION=1
export ENABLE_OPEN_WORLD_TOOLS=1
export TOOL_ACQUISITION_POLICY="${TOOL_ACQUISITION_POLICY:-local_first}"
export ENABLE_MCP_TOOLS="${ENABLE_MCP_TOOLS:-0}"
export MCP_SERVERS_CONFIG="${MCP_SERVERS_CONFIG:-configs/mcp_servers_volcengine.json}"
if [[ "$ENABLE_MCP_TOOLS" == "1" ]] && grep -q '"name": "volcengine_seedance"' "$MCP_SERVERS_CONFIG"; then
  : "${ARK_API_KEY:?Set ARK_API_KEY only when enabling the optional Volcengine MCP fallback}"
fi
export GRAPH_LLM_TIMEOUT_SECONDS="${GRAPH_LLM_TIMEOUT_SECONDS:-300}"
export GRAPH_LLM_MAX_OUTPUT_TOKENS="${GRAPH_LLM_MAX_OUTPUT_TOKENS:-16384}"
export GRAPH_LLM_REPAIR_ATTEMPTS="${GRAPH_LLM_REPAIR_ATTEMPTS:-2}"
export OPEN_WORLD_SYNTHESIS_TIMEOUT_SECONDS="${OPEN_WORLD_SYNTHESIS_TIMEOUT_SECONDS:-300}"
export OPEN_WORLD_SYNTHESIS_MAX_OUTPUT_TOKENS="${OPEN_WORLD_SYNTHESIS_MAX_OUTPUT_TOKENS:-6144}"
export OPEN_WORLD_SYNTHESIS_REPAIR_ATTEMPTS="${OPEN_WORLD_SYNTHESIS_REPAIR_ATTEMPTS:-2}"
export OPEN_WORLD_DISCOVERY_TIMEOUT_SECONDS="${OPEN_WORLD_DISCOVERY_TIMEOUT_SECONDS:-45}"
export OPEN_WORLD_VENV_RETRY_WITHOUT_PROXY="${OPEN_WORLD_VENV_RETRY_WITHOUT_PROXY:-1}"
export OPEN_WORLD_RUNTIME_TIMEOUT_SECONDS="${OPEN_WORLD_RUNTIME_TIMEOUT_SECONDS:-1800}"
export OPEN_WORLD_REQUIRE_SEED_CONTROL="${OPEN_WORLD_REQUIRE_SEED_CONTROL:-1}"
HARNESS_CONFIG="${EVOVIDEO_HARNESS_CONFIG:-configs/open_world_local_wan_harness.json}"

if [[ "$OPEN_WORLD_SANDBOX_BACKEND" != "search-only" && "${OPEN_WORLD_GIT_PREFLIGHT:-1}" == "1" ]]; then
  if [[ -n "${OPEN_WORLD_GIT_CAINFO:-}" ]]; then
    export GIT_SSL_CAINFO="$OPEN_WORLD_GIT_CAINFO"
  fi
  GIT_PREFLIGHT_OUTPUT="$(git ls-remote https://github.com/git/git.git HEAD 2>&1)" || {
    echo "GitHub TLS preflight failed before open-world evolution:" >&2
    echo "$GIT_PREFLIGHT_OUTPUT" >&2
    echo "Configure a PEM bundle containing your organization/proxy root CA:" >&2
    echo "  export OPEN_WORLD_GIT_CAINFO=/path/to/company-ca-bundle.pem" >&2
    echo "Do not disable Git SSL verification." >&2
    exit 1
  }
  echo "GitHub TLS preflight: passed"
fi

PYTHONPATH=src python -m evovideo_skill.cli tool-preflight \
  --config "$HARNESS_CONFIG"

PYTHONPATH=src python -m evovideo_skill.cli run-harness \
  --config "$HARNESS_CONFIG" "$@"
