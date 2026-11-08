#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/jittor-lab}"
CACHE_NAME="${cache_name:-manual}"
CACHE_DIR_NAME="${CACHE_NAME//\//_}"
WORK="$JITTOR_LAB_ROOT/_state/profiling/$CACHE_DIR_NAME"
mkdir -p "$WORK/results" "$WORK/logs"
export JITTOR_LAB_ROOT
export JITTOR_HOME="$WORK/jittor-cache"
export TORCH_HOME="$WORK/torch-cache"
export XDG_CACHE_HOME="$WORK/xdg-cache"
export TMPDIR="$WORK/tmp"
export CUDA_CACHE_PATH="$WORK/cuda-cache"
export cache_name="$CACHE_NAME"
export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Optional CUDA toolkit override (a root with bin/nvcc and lib64). There is no
# default: unset means Jittor discovers nvcc itself; a wrong value is an error.
if [[ -n "${JTCUDA:-}" ]]; then
  if [[ ! -x "$JTCUDA/bin/nvcc" ]]; then
    echo "run_perf_env.sh: JTCUDA=$JTCUDA has no bin/nvcc; point it at a CUDA toolkit root or unset it" >&2
    exit 2
  fi
  export CUDA_HOME="${CUDA_HOME:-$JTCUDA}"
  export nvcc_path="${nvcc_path:-$JTCUDA/bin/nvcc}"
  export PATH="$JTCUDA/bin:$PATH"
  export LD_LIBRARY_PATH="$JTCUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$JITTOR_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$CUDA_CACHE_PATH"

if [[ "${1:-}" == "env" ]]; then
  env | sort | grep -E '^(CUDA[^=]*|JITTOR[^=]*|JTCUDA|LD_LIBRARY_PATH|PYTHONPATH|TORCH_HOME|TMPDIR|XDG_CACHE_HOME|cache_name)='
  exit 0
fi

exec "$@"
