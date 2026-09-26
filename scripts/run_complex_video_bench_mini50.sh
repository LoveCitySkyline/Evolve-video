#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f benchmarks/complex_video_bench_1k/complex_video_bench_mini50.json ]]; then
  PYTHONPATH=src python scripts/generate_complex_video_bench_mini50.py
fi

PYTHONPATH=src "${WAN_PYTHON:-python}" -m evovideo_skill.cli run-harness \
  --config configs/complex_video_bench_mini50.toml
