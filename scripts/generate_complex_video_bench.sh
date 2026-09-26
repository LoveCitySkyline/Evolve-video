#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH=src python -m evovideo_skill.complex_benchmark \
  --output-dir benchmarks/complex_video_bench_1k
