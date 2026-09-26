#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/prepare_complex_video_bench_mini15_tools.sh
exec bash scripts/run_complex_video_bench_mini15_frozen_graph.sh "$@"
