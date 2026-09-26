#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f benchmarks/complex_video_bench_1k/complex_video_bench_mini15.json ]]; then
  PYTHONPATH=src python scripts/generate_complex_video_bench_mini50.py --preset mini15
fi

export EVOVIDEO_HARNESS_CONFIG="configs/open_world_local_wan_mini15_harness.json"
export VIDEO_OUTPUT_DIR="${EVOVIDEO_MINI15_VIDEO_OUTPUT_DIR:-outputs/harness_open_world_local_wan_mini15_videos}"
export EVOVIDEO_AUTO_BOOTSTRAP_ASSETS="${EVOVIDEO_AUTO_BOOTSTRAP_ASSETS:-1}"
export EVOVIDEO_ASSET_BOOTSTRAP_DIR="${EVOVIDEO_ASSET_BOOTSTRAP_DIR:-benchmarks/complex_video_bench_1k/assets/mini15}"
export EVOVIDEO_ASSET_BOOTSTRAP_SEED="${EVOVIDEO_ASSET_BOOTSTRAP_SEED:-20260827}"
export EVOVIDEO_ASSET_BOOTSTRAP_FORCE="${EVOVIDEO_ASSET_BOOTSTRAP_FORCE:-0}"
export EVOVIDEO_MAX_MUTATION_SEARCHES="${EVOVIDEO_MINI15_MUTATION_SEARCHES:-6}"
export OPEN_WORLD_GRAPH_IDEAS="${EVOVIDEO_MINI15_GRAPH_IDEAS:-4}"
export OPEN_WORLD_GRAPH_REALIZATION_BUDGET="${EVOVIDEO_MINI15_REALIZATION_BUDGET:-2}"
export OPEN_WORLD_GRAPH_EXPLORATION_SLOTS="${EVOVIDEO_MINI15_EXPLORATION_SLOTS:-2}"
export OPEN_WORLD_GRAPH_MAX_NODES="${EVOVIDEO_MINI15_GRAPH_MAX_NODES:-8}"
export GRAPH_LLM_MAX_EDITS="${EVOVIDEO_MINI15_GRAPH_MAX_EDITS:-12}"
echo "mini15 isolated video output: $VIDEO_OUTPUT_DIR"
echo "mini15 benchmark assets: auto=$EVOVIDEO_AUTO_BOOTSTRAP_ASSETS dir=$EVOVIDEO_ASSET_BOOTSTRAP_DIR force=$EVOVIDEO_ASSET_BOOTSTRAP_FORCE"
echo "mini15 mutation validation budget: $EVOVIDEO_MAX_MUTATION_SEARCHES graphs total"
echo "mini15 graph invention: ideas=$OPEN_WORLD_GRAPH_IDEAS realization=$OPEN_WORLD_GRAPH_REALIZATION_BUDGET max_nodes=$OPEN_WORLD_GRAPH_MAX_NODES"
exec bash scripts/run_harness_codex_open_world_venv.sh "$@"
