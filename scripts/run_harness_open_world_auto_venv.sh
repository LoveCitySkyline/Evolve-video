#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OPEN_WORLD_SANDBOX_BACKEND="${OPEN_WORLD_SANDBOX_BACKEND:-auto}"
export OPEN_WORLD_VENV_AUTO_APPROVE=1
export OPEN_WORLD_VENV_INTERACTIVE_APPROVAL=0

echo "Open-world venv approval mode: automatic"
echo "  Pinned repositories that pass license, command, path, and security checks"
echo "  will be installed and executed without an interactive confirmation."

bash scripts/run_harness_open_world.sh "$@"
