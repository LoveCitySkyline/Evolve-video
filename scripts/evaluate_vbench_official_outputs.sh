#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${VBENCH_REPO:?Set VBENCH_REPO to the official Vchitect/VBench checkout}"
: "${VBENCH_PYTHON:?Set VBENCH_PYTHON to the Python executable in the VBench environment}"

PROFILE="${VBENCH_PROFILE:-paper-lite}"
OUTPUT_DIR="${EVOVIDEO_VBENCH_OUTPUT_DIR:-outputs/external_vbench_frozen_${PROFILE}}"
RESULTS_DIR="${VBENCH_OFFICIAL_RESULTS_DIR:-$OUTPUT_DIR/official_results}"
INFO_PATH="${VBENCH_INFO_PATH:-$VBENCH_REPO/vbench/VBench_full_info.json}"
FORCE="${VBENCH_OFFICIAL_FORCE:-0}"
MASTER_PORT_BASE="${VBENCH_MASTER_PORT_BASE:-29600}"
ALL_DIMENSIONS="subject_consistency,background_consistency,temporal_flickering,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality,object_class,multiple_objects,human_action,color,spatial_relationship,scene,temporal_style,appearance_style,overall_consistency"

case "$PROFILE" in
  smoke)
    DEFAULT_DIMENSIONS="subject_consistency,background_consistency,motion_smoothness,human_action,overall_consistency"
    ;;
  paper-lite)
    DEFAULT_DIMENSIONS="subject_consistency,background_consistency,temporal_flickering,motion_smoothness,multiple_objects,human_action,spatial_relationship,overall_consistency"
    ;;
  core)
    DEFAULT_DIMENSIONS="subject_consistency,background_consistency,temporal_flickering,motion_smoothness,human_action,overall_consistency"
    ;;
  full)
    DEFAULT_DIMENSIONS="$ALL_DIMENSIONS"
    ;;
  *)
    echo "Unsupported VBENCH_PROFILE=$PROFILE (expected smoke, paper-lite, core, or full)" >&2
    exit 2
    ;;
esac

DIMENSIONS="${VBENCH_DIMENSIONS:-$DEFAULT_DIMENSIONS}"
PROGRAMS="${VBENCH_PROGRAMS:-base,best}"

if [[ ! -f "$VBENCH_REPO/evaluate.py" ]]; then
  echo "Official VBench evaluate.py is missing: $VBENCH_REPO/evaluate.py" >&2
  exit 1
fi
if [[ ! -f "$INFO_PATH" ]]; then
  echo "Official VBench metadata is missing: $INFO_PATH" >&2
  exit 1
fi

if [[ "$VBENCH_PYTHON" == "/path/to/"* ]]; then
  echo "VBENCH_PYTHON is still a placeholder: $VBENCH_PYTHON" >&2
  exit 1
fi
if [[ "$VBENCH_PYTHON" == */* ]]; then
  if [[ ! -x "$VBENCH_PYTHON" ]]; then
    echo "VBENCH_PYTHON is not executable: $VBENCH_PYTHON" >&2
    exit 1
  fi
  VBENCH_PYTHON_BIN="$(cd "$(dirname "$VBENCH_PYTHON")" && pwd)/$(basename "$VBENCH_PYTHON")"
else
  VBENCH_PYTHON_BIN="$(command -v "$VBENCH_PYTHON" || true)"
  if [[ -z "$VBENCH_PYTHON_BIN" ]]; then
    echo "VBENCH_PYTHON was not found on PATH: $VBENCH_PYTHON" >&2
    exit 1
  fi
fi

VBENCH_REPO="$(cd "$VBENCH_REPO" && pwd)"
INFO_PATH="$(cd "$(dirname "$INFO_PATH")" && pwd)/$(basename "$INFO_PATH")"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

if ! (cd "$VBENCH_REPO" && "$VBENCH_PYTHON_BIN" evaluate.py --help >/dev/null); then
  echo "The configured Python cannot load the official VBench evaluator." >&2
  echo "  python=$VBENCH_PYTHON_BIN" >&2
  echo "  repo=$VBENCH_REPO" >&2
  exit 1
fi

if ! "$VBENCH_PYTHON_BIN" -c \
  'import torch; assert torch.cuda.is_available(), "torch.cuda.is_available() is false"; print(torch.cuda.get_device_name(0))' \
  >"$RESULTS_DIR/gpu.txt"; then
  echo "The VBench environment cannot access a CUDA GPU; official VBench requires CUDA." >&2
  exit 1
fi

IFS=',' read -r -a PROGRAM_ARRAY <<< "$PROGRAMS"
IFS=',' read -r -a DIMENSION_ARRAY <<< "$DIMENSIONS"

BASE_COUNT=""
for PROGRAM in "${PROGRAM_ARRAY[@]}"; do
  VIDEO_DIR="$OUTPUT_DIR/official_videos/$PROGRAM"
  if [[ ! -d "$VIDEO_DIR" ]]; then
    echo "Exported $PROGRAM videos are missing: $VIDEO_DIR" >&2
    echo "Run scripts/run_vbench_frozen_graph.sh first with RUN_VBENCH_OFFICIAL=0." >&2
    exit 1
  fi
  VIDEO_COUNT="$(find -L "$VIDEO_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
  if [[ "$VIDEO_COUNT" -eq 0 ]]; then
    echo "No exported MP4 files were found in $VIDEO_DIR" >&2
    exit 1
  fi
  if [[ -z "$BASE_COUNT" ]]; then
    BASE_COUNT="$VIDEO_COUNT"
  elif [[ "$VIDEO_COUNT" -ne "$BASE_COUNT" ]]; then
    echo "Unpaired export: program=$PROGRAM has $VIDEO_COUNT videos, expected $BASE_COUNT" >&2
    exit 1
  fi
done

echo "Official VBench evaluation only (no video generation)"
echo "  profile=$PROFILE"
echo "  programs=$PROGRAMS"
echo "  dimensions=$DIMENSIONS"
echo "  videos_per_program=$BASE_COUNT"
echo "  evaluator_python=$VBENCH_PYTHON_BIN"
echo "  gpu=$(cat "$RESULTS_DIR/gpu.txt")"
echo "  results=$RESULTS_DIR"
if [[ "$PROFILE" != "full" ]]; then
  echo "  protocol=official evaluator on a subset; not the official full leaderboard score"
fi
if [[ "$DIMENSIONS" == *"temporal_flickering"* && "$PROFILE" == "paper-lite" ]]; then
  echo "  warning=paper-lite has one sample per prompt; temporal_flickering is diagnostic, not leaderboard-comparable"
fi

RUN_INDEX=0
for PROGRAM in "${PROGRAM_ARRAY[@]}"; do
  VIDEO_DIR="$(cd "$OUTPUT_DIR/official_videos/$PROGRAM" && pwd)"
  for DIMENSION in "${DIMENSION_ARRAY[@]}"; do
    DIMENSION="${DIMENSION//[[:space:]]/}"
    [[ -n "$DIMENSION" ]] || continue
    RESULT_DIR="$RESULTS_DIR/$PROGRAM/$DIMENSION"
    MARKER="$RESULT_DIR/.complete"
    mkdir -p "$RESULT_DIR"
    if [[ "$FORCE" != "1" && -f "$MARKER" ]]; then
      echo "[VBench official] skip completed program=$PROGRAM dimension=$DIMENSION"
      continue
    fi
    rm -f "$MARKER"
    PORT=$((MASTER_PORT_BASE + RUN_INDEX))
    RUN_INDEX=$((RUN_INDEX + 1))
    echo "[VBench official] start program=$PROGRAM dimension=$DIMENSION port=$PORT"
    (
      cd "$VBENCH_REPO"
      MASTER_ADDR="127.0.0.1" \
      MASTER_PORT="$PORT" \
      RANK=0 \
      LOCAL_RANK=0 \
      WORLD_SIZE=1 \
      "$VBENCH_PYTHON_BIN" evaluate.py \
        --videos_path "$VIDEO_DIR" \
        --dimension "$DIMENSION" \
        --output_path "$RESULT_DIR" \
        --full_json_dir "$INFO_PATH"
    )
    touch "$MARKER"
    echo "[VBench official] done program=$PROGRAM dimension=$DIMENSION"
  done
done

"${PYTHON:-python}" "$ROOT_DIR/scripts/summarize_vbench_official.py" \
  --results-dir "$RESULTS_DIR" \
  --profile "$PROFILE" \
  --programs "$PROGRAMS" \
  --dimensions "$DIMENSIONS"

echo "Official VBench evaluation complete: $RESULTS_DIR"

