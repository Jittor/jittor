#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/gaussian-splatting [entry.py args...]" >&2
  echo "example: $0 /path/to/gaussian-splatting train.py -s DATA -m OUT --iterations 100" >&2
  exit 2
fi

CALLER_CWD="$PWD"
GS_ROOT="$(cd "$1" && pwd)"
shift

if [[ ! -f "$GS_ROOT/train.py" ]]; then
  echo "not a gaussian-splatting checkout: $GS_ROOT" >&2
  exit 2
fi

JT_PY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
JT_PKG_ROOT="$JT_PY_ROOT/jittor"
REAL_HOME="${REAL_HOME:-$HOME}"
CACHE_ROOT="${JITTOR_TORCH_CACHE_ROOT:-${XDG_CACHE_HOME:-$REAL_HOME/.cache}/jittor/torch-shim}"
PROJECT_ID="$(basename "$GS_ROOT")-$(printf '%s' "$GS_ROOT" | sha256sum | cut -c1-16)"
RUNTIME_ROOT="${JITTOR_GS_RUNTIME_ROOT:-$CACHE_ROOT/$PROJECT_ID}"
mkdir -p "$RUNTIME_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  elif [[ -x "$REAL_HOME/miniconda3/envs/jt311/bin/python" ]]; then
    PYTHON_BIN="$REAL_HOME/miniconda3/envs/jt311/bin/python"
  elif [[ -x "$REAL_HOME/anaconda3/envs/jt311/bin/python" ]]; then
    PYTHON_BIN="$REAL_HOME/anaconda3/envs/jt311/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

SHIM_SITE="${SHIM_SITE:-$RUNTIME_ROOT/site-packages}"
export HOME="${JITTOR_GS_HOME:-$RUNTIME_ROOT/home}"
export JITTOR_HOME="${JITTOR_HOME:-$RUNTIME_ROOT/jittor_cache}"
export TORCH_HOME="${TORCH_HOME:-$RUNTIME_ROOT/torch_home}"
export JITTOR_TORCH_EXTENSIONS_DIR="${JITTOR_TORCH_EXTENSIONS_DIR:-$RUNTIME_ROOT/torch_extensions}"
export TMPDIR="${TMPDIR:-$RUNTIME_ROOT/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_ROOT/xdg_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$RUNTIME_ROOT/cuda_cache}"
mkdir -p "$HOME" "$JITTOR_HOME" "$TORCH_HOME" "$JITTOR_TORCH_EXTENSIONS_DIR" "$SHIM_SITE" "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"

JTCUDA_DEFAULT="$REAL_HOME/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux"
if [[ -d "${JTCUDA:-$JTCUDA_DEFAULT}" ]]; then
  JTCUDA="${JTCUDA:-$JTCUDA_DEFAULT}"
  export JTCUDA
  export nvcc_path="${nvcc_path:-$JTCUDA/bin/nvcc}"
  export CUDA_HOME="${CUDA_HOME:-$JTCUDA}"
  export PATH="$JTCUDA/bin:$PATH"
  export LD_LIBRARY_PATH="$JTCUDA/lib64:${LD_LIBRARY_PATH:-}"
fi

export DISABLE_MULTIPROCESSING="${DISABLE_MULTIPROCESSING:-1}"
export use_cutt="${use_cutt:-0}"
export use_cutlass="${use_cutlass:-0}"
export use_nccl="${use_nccl:-0}"
export use_mkl="${use_mkl:-0}"
export JITTOR_TORCH_CUDA_EMPTY_CACHE="${JITTOR_TORCH_CUDA_EMPTY_CACHE:-gc}"
export JITTOR_TORCH_STRICT_BOOTSTRAP="${JITTOR_TORCH_STRICT_BOOTSTRAP:-1}"

export PYTHONPATH="$JT_PY_ROOT:${PYTHONPATH:-}"
echo "[jittor-gs] python: $PYTHON_BIN"
echo "[jittor-gs] runtime: $RUNTIME_ROOT"
echo "[jittor-gs] deploy torch shim: $SHIM_SITE"
"$PYTHON_BIN" "$JT_PKG_ROOT/torch_shim/deploy.py" --target "$SHIM_SITE" >/dev/null

export PYTHONPATH="$SHIM_SITE:$JT_PY_ROOT:$GS_ROOT:$GS_ROOT/submodules/simple-knn:$GS_ROOT/submodules/diff-gaussian-rasterization:$GS_ROOT/submodules/fused-ssim:${PYTHONPATH:-}"
export JITTOR_TORCH_PROJECT_ROOT="${JITTOR_TORCH_PROJECT_ROOT:-$GS_ROOT}"
export JITTOR_TORCH_RUNTIME_ROOT="${JITTOR_TORCH_RUNTIME_ROOT:-$RUNTIME_ROOT}"

for so in $(find "$JITTOR_HOME" -name 'jittor_core*.so' -o -name 'jit_utils_core*.so' 2>/dev/null); do
  dir="$(dirname "$so")"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$dir:"*) ;;
    *) export LD_LIBRARY_PATH="$dir:${LD_LIBRARY_PATH:-}" ;;
  esac
done

build_exts() {
  local dirs=(
    "$GS_ROOT/submodules/simple-knn"
    "$GS_ROOT/submodules/diff-gaussian-rasterization"
    "$GS_ROOT/submodules/fused-ssim"
  )
  for d in "${dirs[@]}"; do
    if [[ -f "$d/setup.py" ]]; then
      echo "[jittor-gs] build_ext: $d"
      (cd "$d" && "$PYTHON_BIN" setup.py build_ext --inplace)
    fi
  done
}

if [[ "${1:-}" == "env" ]]; then
  JITTOR_TORCH_SKIP_EXT_BUILD=1 "$PYTHON_BIN" - <<'PY'
import os, sys
import jittor as torch
torch.flags.torch_shim = 1
import jittor as jt
print("python", sys.executable)
print("jittor", jt.__version__, jt.__file__)
print("torch", getattr(torch, "__version__", None), torch.__file__)
print("cache", jt.flags.cache_path)
print("HOME", os.environ.get("HOME"))
print("JITTOR_HOME", os.environ.get("JITTOR_HOME"))
print("TORCH_HOME", os.environ.get("TORCH_HOME"))
print("JITTOR_TORCH_EXTENSIONS_DIR", os.environ.get("JITTOR_TORCH_EXTENSIONS_DIR"))
print("JITTOR_TORCH_CUDA_EMPTY_CACHE", os.environ.get("JITTOR_TORCH_CUDA_EMPTY_CACHE"))
print("use_cuda", jt.flags.use_cuda)
PY
  exit 0
fi

if [[ "${1:-}" == "build_ext" || "${1:-}" == "build-ext" || "${1:-}" == "extensions" ]]; then
  build_exts
  exit 0
fi

if [[ "${JITTOR_GS_SKIP_EXT_BUILD:-0}" != "1" ]]; then
  build_exts
fi
# The explicit builds above are fail-fast and use the same generic Jittor
# cpp_extension backend. Avoid launching a second setup.py scan when Jittor
# bootstraps the entrypoint.
export JITTOR_TORCH_SKIP_EXT_BUILD="${JITTOR_TORCH_SKIP_EXT_BUILD:-1}"

if [[ $# -eq 0 ]]; then
  set -- train.py
fi

ENTRY="$1"
shift
if [[ "$ENTRY" != /* ]]; then
  if [[ -f "$GS_ROOT/$ENTRY" ]]; then
    ENTRY="$GS_ROOT/$ENTRY"
  else
    ENTRY="$CALLER_CWD/$ENTRY"
  fi
fi
if [[ ! -f "$ENTRY" ]]; then
  echo "python entrypoint not found: $ENTRY" >&2
  exit 2
fi
if [[ "$(basename "$ENTRY")" == "metrics.py" ]]; then
  export JITTOR_TORCH_INFERENCE="${JITTOR_TORCH_INFERENCE:-1}"
fi

cd "$GS_ROOT"
exec "$PYTHON_BIN" -c '
import os
import runpy
import sys

entry = os.path.abspath(sys.argv[1])
sys.argv = [entry, *sys.argv[2:]]
import jittor as torch
torch.flags.torch_shim = 1
runpy.run_path(entry, run_name="__main__")
' "$ENTRY" "$@"
