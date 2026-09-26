#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${WAN_REPO:?Set WAN_REPO to the local Wan repository containing generate.py}"
: "${WAN_CKPT_DIR:?Set WAN_CKPT_DIR to the local Wan T2V checkpoint directory}"
: "${STORYBENCH_REPO:?Set STORYBENCH_REPO to the official google/storybench checkout}"

PROFILE="${STORYBENCH_PROFILE:-smoke}"
MINI50_RUN_DIR="${EVOVIDEO_MINI50_RUN_DIR:-outputs/harness_frozen_tools_local_wan_mini50/complex_video_bench_mini50_full}"
SOURCE_TASKS="${STORYBENCH_TASK_PATH:-$STORYBENCH_REPO/data/tasks/oops-test/story_gen.json}"
TASK_PATH="benchmarks/external/storybench_story_gen.json"
OUTPUT_DIR="${EVOVIDEO_STORYBENCH_OUTPUT_DIR:-outputs/external_storybench_frozen_${PROFILE}}"
CONFIG="configs/external_storybench_frozen_harness.json"

case "$PROFILE" in
  smoke) DEFAULT_LIMIT=2 ;;
  paper-lite) DEFAULT_LIMIT=24 ;;
  full) DEFAULT_LIMIT="" ;;
  *)
    echo "Unsupported STORYBENCH_PROFILE=$PROFILE (expected smoke, paper-lite, or full)" >&2
    exit 2
    ;;
esac
LIMIT="${STORYBENCH_LIMIT:-$DEFAULT_LIMIT}"
if [[ "$PROFILE" == "paper-lite" ]]; then
  DEFAULT_SAMPLES=1
else
  DEFAULT_SAMPLES=4
fi
SAMPLES="${STORYBENCH_SAMPLES_PER_PROMPT:-$DEFAULT_SAMPLES}"

if [[ ! -f "$SOURCE_TASKS" ]]; then
  echo "StoryBench annotations are missing: $SOURCE_TASKS" >&2
  exit 1
fi
if [[ ! -d "$MINI50_RUN_DIR/evolution/registry" ]]; then
  echo "Completed mini50 run is missing: $MINI50_RUN_DIR" >&2
  exit 1
fi

PREPARE_ARGS=(
  storybench
  --tasks "$SOURCE_TASKS"
  --output "$TASK_PATH"
  --task-mode story_gen
  --samples-per-prompt "$SAMPLES"
)
if [[ -n "$LIMIT" ]]; then
  PREPARE_ARGS+=(--limit "$LIMIT")
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
export WAN_TIMEOUT_SECONDS="${WAN_TIMEOUT_SECONDS:-1800}"
export EVOVIDEO_BASELINE_CONDITIONED_REPAIR="${EVOVIDEO_BASELINE_CONDITIONED_REPAIR:-1}"
export EVOVIDEO_RUNTIME_REPAIR_MIN_GAIN="${EVOVIDEO_RUNTIME_REPAIR_MIN_GAIN:-0.02}"

echo "Frozen mini50 -> StoryBench story_gen transfer"
echo "  profile=$PROFILE examples=${LIMIT:-all} samples/example=$SAMPLES"
echo "  source_frontier=$MINI50_RUN_DIR"
echo "  output=$OUTPUT_DIR"
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
    --benchmark storybench \
    --tasks "$TASK_PATH" \
    --report "$OUTPUT_DIR/external_eval.json" \
    --program "$PROGRAM" \
    --output-dir "$OUTPUT_DIR/official_data/$PROGRAM/story_gen/oops_test/raw"
done

echo "StoryBench generation/export complete: $OUTPUT_DIR"
echo "Run the archived official metric modules from STORYBENCH_REPO after installing their checkpoints."
