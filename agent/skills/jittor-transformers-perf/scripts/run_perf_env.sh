#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/jittor-lab}"
WORK="${JITTOR_TRANSFORMERS_PERF_WORKDIR:-$JITTOR_LAB_ROOT/jittor_transformers_perf}"
RUNTIME="$WORK/runtime"
CACHE_NAME="${cache_name:-jittor_transformers_perf}"
CACHE_DIR_NAME="${CACHE_NAME//\//_}"

mkdir -p "$RUNTIME" "$WORK/results" "$WORK/logs"

# The caller's own home, captured before HOME is redirected below.
export REAL_HOME="${REAL_HOME:-$HOME}"
export JITTOR_TORCH_PROJECT_ROOT="$WORK"
export JITTOR_TORCH_RUNTIME_ROOT="$RUNTIME/$CACHE_DIR_NAME/jittor"
export JITTOR_HOME="$RUNTIME/$CACHE_DIR_NAME/jittor_home"
export HOME="$RUNTIME/$CACHE_DIR_NAME/home"
export TORCH_HOME="$RUNTIME/$CACHE_DIR_NAME/torch_home"
export XDG_CACHE_HOME="$RUNTIME/$CACHE_DIR_NAME/xdg_cache"
export TMPDIR="$RUNTIME/$CACHE_DIR_NAME/tmp"
export CUDA_CACHE_PATH="$RUNTIME/$CACHE_DIR_NAME/cuda_cache"
export cache_name="$CACHE_NAME"
export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUNTIME/hf_home"
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

mkdir -p "$HOME" "$JITTOR_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$CUDA_CACHE_PATH"

if [[ "${1:-}" == "env" ]]; then
  env | sort | grep -E '^(CUDA[^=]*|JITTOR[^=]*|JTCUDA|LD_LIBRARY_PATH|PYTHONPATH|REAL_HOME|HOME|TORCH_HOME|TMPDIR|XDG_CACHE_HOME|cache_name)='
  exit 0
fi

exec "$@"
