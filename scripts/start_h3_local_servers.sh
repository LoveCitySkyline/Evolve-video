#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${H3_MODEL_PATH:-MiniMaxAI/MiniMax-H3}"
LOG_DIR="${H3_SERVER_LOG_DIR:-$ROOT_DIR/outputs/h3_local_servers}"

FL2VA_GPUS="${H3_FL2VA_GPUS:-0,1,2,3}"
REF2VA_GPUS="${H3_REF2VA_GPUS:-4,5,6,7}"

# Isolate all distributed-control ports. Concurrent SGLang diffusion servers
# otherwise race for the default master port (30005).
FL2VA_HTTP_PORT="${H3_FL2VA_PORT:-30010}"
REF2VA_HTTP_PORT="${H3_REF2VA_PORT:-30011}"
FL2VA_MASTER_PORT="${H3_FL2VA_MASTER_PORT:-30105}"
REF2VA_MASTER_PORT="${H3_REF2VA_MASTER_PORT:-30205}"
FL2VA_SCHEDULER_PORT="${H3_FL2VA_SCHEDULER_PORT:-30110}"
REF2VA_SCHEDULER_PORT="${H3_REF2VA_SCHEDULER_PORT:-30210}"

NUM_GPUS="${H3_NUM_GPUS_PER_SERVER:-4}"
# Official lossless topology for four 80 GB H100s. TP > 1 requires a CUDA 12.x
# nvcc because SGLang builds its custom all-reduce kernel on first launch.
TP_SIZE="${H3_TP_SIZE:-2}"
ULYSSES_DEGREE="${H3_ULYSSES_DEGREE:-2}"
ENCODER_PARALLEL="${H3_ENCODER_PARALLEL:-auto}"
PERFORMANCE_MODE="${H3_PERFORMANCE_MODE:-speed}"
HOST="${H3_SERVER_HOST:-127.0.0.1}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

SGLANG_BIN="${H3_SGLANG_BIN:-sglang}"
if (( ! DRY_RUN )); then
  if [[ "$SGLANG_BIN" == */* ]]; then
    if [[ ! -x "$SGLANG_BIN" ]]; then
      echo "SGLang executable not found or not executable: $SGLANG_BIN" >&2
      exit 127
    fi
    SGLANG_BIN="$(cd "$(dirname "$SGLANG_BIN")" && pwd)/$(basename "$SGLANG_BIN")"
  else
    if ! SGLANG_BIN="$(command -v "$SGLANG_BIN")"; then
      echo "SGLang executable not found: $SGLANG_BIN. Activate h3-sglang-cu129 or set H3_SGLANG_BIN." >&2
      exit 127
    fi
  fi
fi

NVCC_BIN="${H3_NVCC_BIN:-}"
if [[ ! "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "H3_TP_SIZE must be a positive integer, got: $TP_SIZE" >&2
  exit 2
fi
if (( ! DRY_RUN && TP_SIZE > 1 )) && [[ "${H3_SKIP_NVCC_CHECK:-0}" != "1" ]]; then
  if [[ -z "$NVCC_BIN" && -n "${H3_CUDA_HOME:-}" ]]; then
    NVCC_BIN="$H3_CUDA_HOME/bin/nvcc"
  fi
  if [[ -z "$NVCC_BIN" && -n "${CUDA_HOME:-}" ]]; then
    NVCC_BIN="$CUDA_HOME/bin/nvcc"
  fi
  if [[ -z "$NVCC_BIN" && -x /usr/local/cuda/bin/nvcc ]]; then
    NVCC_BIN=/usr/local/cuda/bin/nvcc
  fi
  if [[ -z "$NVCC_BIN" ]]; then
    NVCC_BIN="$(command -v nvcc || true)"
  fi
  if [[ -z "$NVCC_BIN" || ! -x "$NVCC_BIN" ]]; then
    echo "H3_TP_SIZE=$TP_SIZE enables the SGLang custom all-reduce JIT, which requires nvcc." >&2
    echo "Set H3_NVCC_BIN=/path/to/a/CUDA-12.x/bin/nvcc or H3_CUDA_HOME=/path/to/cuda." >&2
    exit 2
  fi
  # `nvcc --help` does not consistently enumerate every accepted C++ dialect.
  # Compile a tiny translation unit instead of inferring support from help text.
  NVCC_PROBE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/evovideo-nvcc.XXXXXX")"
  printf '%s\n' '__global__ void probe() {}' >"$NVCC_PROBE_DIR/probe.cu"
  if ! "$NVCC_BIN" -std=c++20 -c "$NVCC_PROBE_DIR/probe.cu" \
      -o "$NVCC_PROBE_DIR/probe.o" \
      >"$NVCC_PROBE_DIR/stdout.log" 2>"$NVCC_PROBE_DIR/stderr.log"; then
    echo "The selected nvcc failed a real -std=c++20 compile probe: $NVCC_BIN" >&2
    "$NVCC_BIN" --version >&2 || true
    cat "$NVCC_PROBE_DIR/stderr.log" >&2 || true
    rm -rf "$NVCC_PROBE_DIR"
    echo "Set H3_NVCC_BIN to a working CUDA 12.x compiler, or use H3_TP_SIZE=1 H3_ULYSSES_DEGREE=4." >&2
    exit 2
  fi
  rm -rf "$NVCC_PROBE_DIR"
  NVCC_BIN="$(cd "$(dirname "$NVCC_BIN")" && pwd)/$(basename "$NVCC_BIN")"
  export CUDACXX="$NVCC_BIN"
  export CUDA_HOME="${H3_CUDA_HOME:-$(cd "$(dirname "$NVCC_BIN")/.." && pwd)}"
  export PATH="$(dirname "$NVCC_BIN"):$PATH"
fi

# SGLang 0.5.19 recognizes a local MiniMax H3 checkpoint by its canonical
# basename. Normalize arbitrary directory names so detection cannot fall back
# to the generic Diffusers pipeline.
ORIGINAL_MODEL_PATH="$MODEL_PATH"
if (( ! DRY_RUN )) && [[ -d "$MODEL_PATH" ]]; then
  MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
  for required_file in \
    model_index.json \
    FL2VA/model_index.json \
    Ref2VA/model_index.json; do
    if [[ ! -f "$MODEL_PATH/$required_file" ]]; then
      echo "Incomplete MiniMax H3 checkpoint: missing $MODEL_PATH/$required_file" >&2
      exit 2
    fi
  done

  if [[ "$(basename "$MODEL_PATH")" != "MiniMax-H3" ]]; then
    MODEL_ALIAS_DIR="$LOG_DIR/model_alias"
    MODEL_ALIAS_PATH="$MODEL_ALIAS_DIR/MiniMax-H3"
    mkdir -p "$MODEL_ALIAS_DIR"
    if [[ -e "$MODEL_ALIAS_PATH" && ! -L "$MODEL_ALIAS_PATH" ]]; then
      echo "Cannot create H3 model alias: $MODEL_ALIAS_PATH already exists and is not a symlink" >&2
      exit 2
    fi
    ln -sfn "$MODEL_PATH" "$MODEL_ALIAS_PATH"
    MODEL_PATH="$MODEL_ALIAS_PATH"
  fi
fi

# Check native H3 support before reserving and initializing all eight GPUs.
SGLANG_PYTHON="${H3_SGLANG_PYTHON:-$(dirname "$SGLANG_BIN")/python}"
if (( ! DRY_RUN )) && [[ -x "$SGLANG_PYTHON" ]] && [[ "${H3_SKIP_PIPELINE_CHECK:-0}" != "1" ]]; then
  if ! "$SGLANG_PYTHON" -c \
    'from sglang.multimodal_gen.registry import get_pipeline_class; assert get_pipeline_class("MiniMaxH3Pipeline") is not None, "MiniMaxH3Pipeline is not registered"'; then
    echo "The selected SGLang environment does not provide the native MiniMaxH3Pipeline." >&2
    echo "Check H3_SGLANG_BIN/H3_SGLANG_PYTHON and reinstall an H3-capable SGLang release." >&2
    exit 2
  fi
fi

# CUDA 12.4 can reject the ADL-style tvm::ffi::get calls used by the SGLang
# CUDA-IPC JIT source even though the equivalent Tuple member API compiles.
# Apply the semantics-preserving compatibility edit once in the dedicated H3
# environment, then discard only this module's stale JIT cache.
H3_PATCH_SGLANG_IPC="${H3_PATCH_SGLANG_IPC:-auto}"
if [[ "$H3_PATCH_SGLANG_IPC" != "auto" && "$H3_PATCH_SGLANG_IPC" != "0" && "$H3_PATCH_SGLANG_IPC" != "1" ]]; then
  echo "H3_PATCH_SGLANG_IPC must be auto, 0, or 1; got: $H3_PATCH_SGLANG_IPC" >&2
  exit 2
fi
if (( ! DRY_RUN && TP_SIZE > 1 )) && [[ -x "$SGLANG_PYTHON" ]] && [[ "$H3_PATCH_SGLANG_IPC" != "0" ]]; then
  SGLANG_PACKAGE_DIR="$("$SGLANG_PYTHON" -c \
    'from pathlib import Path; import sglang; print(Path(sglang.__file__).resolve().parent)')"
  IPC_HEADER="$SGLANG_PACKAGE_DIR/kernels/jit/csrc/distributed/ipc.cuh"
  if [[ -f "$IPC_HEADER" ]] && grep -Fq 'get<0>(pair)' "$IPC_HEADER"; then
    IPC_BACKUP="$IPC_HEADER.evovideo-original"
    if [[ ! -f "$IPC_BACKUP" ]]; then
      cp -p "$IPC_HEADER" "$IPC_BACKUP"
    fi
    "$SGLANG_PYTHON" - "$IPC_HEADER" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
replacements = {
    "get<0>(pair)": "pair.get<0>()",
    "get<1>(pair)": "pair.get<1>()",
}
for old, new in replacements.items():
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"expected one {old!r} in {path}, found {count}")
    source = source.replace(old, new)
path.write_text(source)
PY
    echo "Applied CUDA 12.4 SGLang IPC compatibility patch: $IPC_HEADER" >&2

    SGLANG_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/sglang/jit"
    if [[ -d "$SGLANG_CACHE_ROOT" ]]; then
      while IFS= read -r -d '' cache_dir; do
        rm -rf "$cache_dir"
      done < <(find "$SGLANG_CACHE_ROOT" -mindepth 2 -maxdepth 2 \
        -type d -name sgl_kernel_jit_cuda_ipc -print0 2>/dev/null)
    fi
  elif [[ "$H3_PATCH_SGLANG_IPC" == "1" ]] && [[ ! -f "$IPC_HEADER" ]]; then
    echo "Cannot patch SGLang CUDA IPC source; file not found: $IPC_HEADER" >&2
    exit 2
  fi
fi

validate_port() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1024 || value > 65535 )); then
    echo "$name must be an integer in [1024, 65535], got: $value" >&2
    exit 2
  fi
}

declare -a PORT_NAMES=(
  FL2VA_HTTP_PORT REF2VA_HTTP_PORT
  FL2VA_MASTER_PORT REF2VA_MASTER_PORT
  FL2VA_SCHEDULER_PORT REF2VA_SCHEDULER_PORT
)
declare -a PORT_VALUES=(
  "$FL2VA_HTTP_PORT" "$REF2VA_HTTP_PORT"
  "$FL2VA_MASTER_PORT" "$REF2VA_MASTER_PORT"
  "$FL2VA_SCHEDULER_PORT" "$REF2VA_SCHEDULER_PORT"
)

for index in "${!PORT_VALUES[@]}"; do
  validate_port "${PORT_NAMES[$index]}" "${PORT_VALUES[$index]}"
done
for ((i = 0; i < ${#PORT_VALUES[@]}; i++)); do
  for ((j = i + 1; j < ${#PORT_VALUES[@]}; j++)); do
    if [[ "${PORT_VALUES[$i]}" == "${PORT_VALUES[$j]}" ]]; then
      echo "Port collision: ${PORT_NAMES[$i]} and ${PORT_NAMES[$j]} both use ${PORT_VALUES[$i]}" >&2
      exit 2
    fi
  done
done

COMMON_ARGS=(
  serve
  --model-path "$MODEL_PATH"
  --model-type diffusion
  --backend sglang
  --model-id MiniMax-H3
  --pipeline-class-name MiniMaxH3Pipeline
  --num-gpus "$NUM_GPUS"
  --tp-size "$TP_SIZE"
  --ulysses-degree "$ULYSSES_DEGREE"
  --encoder-parallel "$ENCODER_PARALLEL"
  --performance-mode "$PERFORMANCE_MODE"
  --host "$HOST"
  --strict-ports
)

FL2VA_ARGS=(
  "${COMMON_ARGS[@]}"
  --model-variant fl2va
  --port "$FL2VA_HTTP_PORT"
  --master-port "$FL2VA_MASTER_PORT"
  --scheduler-port "$FL2VA_SCHEDULER_PORT"
)
REF2VA_ARGS=(
  "${COMMON_ARGS[@]}"
  --model-variant ref2va
  --port "$REF2VA_HTTP_PORT"
  --master-port "$REF2VA_MASTER_PORT"
  --scheduler-port "$REF2VA_SCHEDULER_PORT"
)

print_command() {
  local devices="$1"
  shift
  printf 'CUDA_VISIBLE_DEVICES=%q %q' "$devices" "$SGLANG_BIN"
  printf ' %q' "$@"
  printf '\n'
}

echo "Starting isolated H3 SGLang services" >&2
echo "  executable=$SGLANG_BIN" >&2
echo "  model=$MODEL_PATH" >&2
if [[ "$ORIGINAL_MODEL_PATH" != "$MODEL_PATH" ]]; then
  echo "  model_source=$ORIGINAL_MODEL_PATH" >&2
fi
if [[ -n "$NVCC_BIN" ]]; then
  echo "  nvcc=$NVCC_BIN" >&2
fi
echo "  FL2VA GPUs=$FL2VA_GPUS HTTP=http://$HOST:$FL2VA_HTTP_PORT master=$FL2VA_MASTER_PORT scheduler=$FL2VA_SCHEDULER_PORT" >&2
echo "  Ref2VA GPUs=$REF2VA_GPUS HTTP=http://$HOST:$REF2VA_HTTP_PORT master=$REF2VA_MASTER_PORT scheduler=$REF2VA_SCHEDULER_PORT" >&2

if (( DRY_RUN )); then
  print_command "$FL2VA_GPUS" "${FL2VA_ARGS[@]}"
  print_command "$REF2VA_GPUS" "${REF2VA_ARGS[@]}"
  exit 0
fi

mkdir -p "$LOG_DIR"
FL2VA_LOG="$LOG_DIR/fl2va.log"
REF2VA_LOG="$LOG_DIR/ref2va.log"
ARCHIVE_SUFFIX="$(date -u +%Y%m%dT%H%M%SZ)-$$"
for log_file in "$FL2VA_LOG" "$REF2VA_LOG"; do
  if [[ -s "$log_file" ]]; then
    mv "$log_file" "${log_file%.log}.$ARCHIVE_SUFFIX.log"
  fi
done

echo "  logs=$FL2VA_LOG,$REF2VA_LOG"
echo "  loading is not a readiness check; follow the logs until both HTTP servers are ready"

cleanup() {
  trap - INT TERM EXIT
  for pid in "${FL2VA_PID:-}" "${REF2VA_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait "${FL2VA_PID:-}" "${REF2VA_PID:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

CUDA_VISIBLE_DEVICES="$FL2VA_GPUS" "$SGLANG_BIN" "${FL2VA_ARGS[@]}" >"$FL2VA_LOG" 2>&1 &
FL2VA_PID=$!
CUDA_VISIBLE_DEVICES="$REF2VA_GPUS" "$SGLANG_BIN" "${REF2VA_ARGS[@]}" >"$REF2VA_LOG" 2>&1 &
REF2VA_PID=$!

echo "  FL2VA pid=$FL2VA_PID"
echo "  Ref2VA pid=$REF2VA_PID"
echo "Keep this supervisor running; Ctrl-C stops both services."

while true; do
  if ! kill -0 "$FL2VA_PID" 2>/dev/null; then
    status=0
    wait "$FL2VA_PID" || status=$?
    echo "FL2VA SGLang process $FL2VA_PID exited (status $status); stopping both services." >&2
    (( status == 0 )) && status=1
    exit "$status"
  fi
  if ! kill -0 "$REF2VA_PID" 2>/dev/null; then
    status=0
    wait "$REF2VA_PID" || status=$?
    echo "Ref2VA SGLang process $REF2VA_PID exited (status $status); stopping both services." >&2
    (( status == 0 )) && status=1
    exit "$status"
  fi
  sleep 2
done
