#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OPEN_WORLD_SANDBOX_BACKEND="${OPEN_WORLD_SANDBOX_BACKEND:-auto}"
: "${OPEN_WORLD_VENV_APPROVAL_FILE:=$ROOT_DIR/configs/open_world_venv_approvals.json}"
export OPEN_WORLD_VENV_APPROVAL_FILE
export OPEN_WORLD_VENV_INTERACTIVE_APPROVAL="${OPEN_WORLD_VENV_INTERACTIVE_APPROVAL:-1}"

echo "Open-world venv approval mode: interactive"
echo "  approval_file=$OPEN_WORLD_VENV_APPROVAL_FILE"
echo "  Unapproved repositories will be shown before host-side install or execution."

bash scripts/run_harness_open_world.sh "$@"

PYTHONPATH=src python -m evovideo_skill.cli tool-approvals \
  --config configs/open_world_local_wan_harness.json \
  --approval-file "$OPEN_WORLD_VENV_APPROVAL_FILE"
