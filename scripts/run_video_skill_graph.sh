#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROVIDER="${PROVIDER:-local-fake}"
MEMORY_DIR="${MEMORY_DIR:-outputs/video_skill_graph_memory}"
TASKS="${TASKS:-examples/graph_evolution_tasks.json}"
LIMIT_TASKS="${LIMIT_TASKS:-4}"
MAX_CANDIDATES="${MAX_CANDIDATES:-8}"
MIN_QUALITY_GAIN="${MIN_QUALITY_GAIN:-0.02}"
AUTO_CONSOLIDATE_FOUNDATION="${AUTO_CONSOLIDATE_FOUNDATION:-1}"
MIN_SUPPORT="${MIN_SUPPORT:-2}"
MIN_UTILITY="${MIN_UTILITY:-0.02}"
MAX_MOTIF_SIZE="${MAX_MOTIF_SIZE:-4}"
MAX_FOUNDATION_SKILLS="${MAX_FOUNDATION_SKILLS:-5}"

ARGS=(
  graph-evolve
  --provider "$PROVIDER"
  --memory-dir "$MEMORY_DIR"
  --tasks "$TASKS"
  --limit-tasks "$LIMIT_TASKS"
  --max-candidates "$MAX_CANDIDATES"
  --min-quality-gain "$MIN_QUALITY_GAIN"
)

if [[ "$AUTO_CONSOLIDATE_FOUNDATION" == "1" ]]; then
  ARGS+=(
    --auto-consolidate-foundation
    --min-support "$MIN_SUPPORT"
    --min-utility "$MIN_UTILITY"
    --max-motif-size "$MAX_MOTIF_SIZE"
    --max-foundation-skills "$MAX_FOUNDATION_SKILLS"
  )
fi

if [[ "${ENABLE_VLM_EVAL:-0}" == "1" ]]; then
  ARGS+=(--enable-vlm-eval --vlm-model "${VLM_MODEL:-qwen3-vl-plus}")
fi

if [[ "$PROVIDER" == "local-wan" ]]; then
  ARGS+=(
    --wan-repo "${WAN_REPO:?WAN_REPO is required for PROVIDER=local-wan}"
    --wan-ckpt-dir "${WAN_CKPT_DIR:?WAN_CKPT_DIR is required for PROVIDER=local-wan}"
    --wan-task "${WAN_TASK:-t2v-1.3B}"
    --wan-size "${WAN_SIZE:-832*480}"
  )
fi

PYTHONPATH=src python -m evovideo_skill.cli "${ARGS[@]}"
