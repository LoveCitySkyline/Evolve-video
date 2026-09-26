#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${WAN_REPO:?Set WAN_REPO to the local Wan repository containing generate.py}"
: "${WAN_CKPT_DIR:?Set WAN_CKPT_DIR to the local Wan T2V checkpoint directory}"
: "${VBENCH_REPO:?Set VBENCH_REPO to the official Vchitect/VBench checkout}"

PROFILE="${VBENCH_PROFILE:-smoke}"
MINI50_RUN_DIR="${EVOVIDEO_MINI50_RUN_DIR:-outputs/harness_frozen_tools_local_wan_mini50/complex_video_bench_mini50_full}"
INFO_PATH="${VBENCH_INFO_PATH:-$VBENCH_REPO/vbench/VBench_full_info.json}"
TASK_PATH="benchmarks/external/vbench_official.json"
OUTPUT_DIR="${EVOVIDEO_VBENCH_OUTPUT_DIR:-outputs/external_vbench_frozen_${PROFILE}}"
CONFIG="configs/external_vbench_frozen_harness.json"
ALL_DIMENSIONS="subject_consistency,background_consistency,temporal_flickering,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality,object_class,multiple_objects,human_action,color,spatial_relationship,scene,temporal_style,appearance_style,overall_consistency"

case "$PROFILE" in
  smoke)
    DEFAULT_DIMENSIONS="subject_consistency,background_consistency,motion_smoothness,human_action,overall_consistency"
    DEFAULT_LIMIT=1
    DEFAULT_SAMPLES=1
    DEFAULT_FLICKER_SAMPLES=1
    ;;
  paper-lite)
    DEFAULT_DIMENSIONS="subject_consistency,background_consistency,temporal_flickering,motion_smoothness,multiple_objects,human_action,spatial_relationship,overall_consistency"
    DEFAULT_LIMIT=8
    DEFAULT_SAMPLES=1
    DEFAULT_FLICKER_SAMPLES=1
    ;;
  core)
    DEFAULT_DIMENSIONS="subject_consistency,background_consistency,temporal_flickering,motion_smoothness,human_action,overall_consistency"
    DEFAULT_LIMIT=""
    DEFAULT_SAMPLES=5
    DEFAULT_FLICKER_SAMPLES=25
    ;;
  full)
    DEFAULT_DIMENSIONS="$ALL_DIMENSIONS"
    DEFAULT_LIMIT=""
    DEFAULT_SAMPLES=5
    DEFAULT_FLICKER_SAMPLES=25
    ;;
  *)
    echo "Unsupported VBENCH_PROFILE=$PROFILE (expected smoke, paper-lite, core, or full)" >&2
    exit 2
    ;;
esac

DIMENSIONS="${VBENCH_DIMENSIONS:-$DEFAULT_DIMENSIONS}"
LIMIT="${VBENCH_LIMIT_PROMPTS_PER_DIMENSION:-$DEFAULT_LIMIT}"
SAMPLES="${VBENCH_SAMPLES_PER_PROMPT:-$DEFAULT_SAMPLES}"
FLICKER_SAMPLES="${VBENCH_TEMPORAL_FLICKERING_SAMPLES:-$DEFAULT_FLICKER_SAMPLES}"

if [[ ! -f "$INFO_PATH" ]]; then
  echo "VBench prompt metadata is missing: $INFO_PATH" >&2
  exit 1
fi
if [[ ! -d "$MINI50_RUN_DIR/evolution/registry" ]]; then
  echo "Completed mini50 run is missing: $MINI50_RUN_DIR" >&2
  exit 1
fi

if [[ "${RUN_VBENCH_OFFICIAL:-0}" == "1" ]]; then
  VBENCH_PYTHON_BIN="${VBENCH_PYTHON:-python}"
  if [[ "$VBENCH_PYTHON_BIN" == "/path/to/"* ]]; then
    echo "VBENCH_PYTHON is still a placeholder: $VBENCH_PYTHON_BIN" >&2
    echo "Set it to the executable in your installed VBench environment." >&2
    exit 1
  fi
  if [[ "$VBENCH_PYTHON_BIN" == */* ]]; then
    if [[ ! -x "$VBENCH_PYTHON_BIN" ]]; then
      echo "VBENCH_PYTHON is not an executable file: $VBENCH_PYTHON_BIN" >&2
      exit 1
    fi
    VBENCH_PYTHON_BIN="$(cd "$(dirname "$VBENCH_PYTHON_BIN")" && pwd)/$(basename "$VBENCH_PYTHON_BIN")"
  else
    VBENCH_PYTHON_BIN="$(command -v "$VBENCH_PYTHON_BIN" || true)"
    if [[ -z "$VBENCH_PYTHON_BIN" ]]; then
      echo "VBENCH_PYTHON was not found on PATH." >&2
      exit 1
    fi
  fi
  if ! (cd "$VBENCH_REPO" && "$VBENCH_PYTHON_BIN" evaluate.py --help >/dev/null); then
    echo "The configured VBench Python cannot load $VBENCH_REPO/evaluate.py" >&2
    echo "Install the official VBench dependencies in that environment before generation." >&2
    exit 1
  fi
fi

PREPARE_ARGS=(
  vbench
  --info "$INFO_PATH"
  --output "$TASK_PATH"
  --dimensions "$DIMENSIONS"
  --samples-per-prompt "$SAMPLES"
  --temporal-flickering-samples "$FLICKER_SAMPLES"
)
if [[ -n "$LIMIT" ]]; then
  PREPARE_ARGS+=(--limit-prompts-per-dimension "$LIMIT")
fi

export PROVIDER=local-wan
export RESTORE_CATALOG_TOOLS=1
export ENABLE_OPEN_WORLD_TOOLS=0
export ENABLE_MCP_TOOLS=0
export ENABLE_LLM_MUTATION=0
export ENABLE_VLM_EVAL="${ENABLE_VLM_EVAL:-1}"
if [[ "$ENABLE_VLM_EVAL" == "1" ]]; then
  : "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY for verifier-gated runtime repair}"
fi
export TOOL_CATALOG_PATH="${TOOL_CATALOG_PATH:-${EVOVIDEO_PREPARED_TOOL_STORE:-outputs/prepared_tool_store_mini15}/tool_catalog.json}"
export VIDEO_OUTPUT_DIR="$OUTPUT_DIR/videos"
export EVOVIDEO_AGENT_STATE_DIR="$OUTPUT_DIR/agent_state"
export WAN_TIMEOUT_SECONDS="${WAN_TIMEOUT_SECONDS:-900}"
export EVOVIDEO_BASELINE_CONDITIONED_REPAIR="${EVOVIDEO_BASELINE_CONDITIONED_REPAIR:-1}"
export EVOVIDEO_RUNTIME_REPAIR_MIN_GAIN="${EVOVIDEO_RUNTIME_REPAIR_MIN_GAIN:-0.02}"

echo "Frozen mini50 -> VBench transfer"
echo "  profile=$PROFILE dimensions=$DIMENSIONS"
echo "  source_frontier=$MINI50_RUN_DIR"
echo "  output=$OUTPUT_DIR"
echo "  official_eval=${RUN_VBENCH_OFFICIAL:-0}"
echo "  verifier_gated_repair=1 min_gain=$EVOVIDEO_RUNTIME_REPAIR_MIN_GAIN"

PYTHONPATH=src "${PYTHON:-python}" scripts/prepare_external_benchmark.py "${PREPARE_ARGS[@]}"
PYTHONPATH=src "${PYTHON:-python}" -m evovideo_skill.cli eval-external \
  --config "$CONFIG" \
  --source-run-dir "$MINI50_RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --require-specialized-routing \
  "$@"

for PROGRAM in base best; do
  PYTHONPATH=src "${PYTHON:-python}" scripts/export_external_benchmark_outputs.py \
    --benchmark vbench \
    --tasks "$TASK_PATH" \
    --report "$OUTPUT_DIR/external_eval.json" \
    --program "$PROGRAM" \
    --output-dir "$OUTPUT_DIR/official_videos/$PROGRAM"
done

if [[ "${RUN_VBENCH_OFFICIAL:-0}" == "1" ]]; then
  VBENCH_PROFILE="$PROFILE" \
  VBENCH_DIMENSIONS="$DIMENSIONS" \
  VBENCH_INFO_PATH="$INFO_PATH" \
  VBENCH_PYTHON="$VBENCH_PYTHON_BIN" \
  EVOVIDEO_VBENCH_OUTPUT_DIR="$OUTPUT_DIR" \
    bash scripts/evaluate_vbench_official_outputs.sh
fi

echo "VBench transfer complete: $OUTPUT_DIR"
