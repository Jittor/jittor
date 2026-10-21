#!/usr/bin/env bash
# Drive a full jittor<->real-torch forward+backward parity run across both envs.
# Usage: run_parity.sh <arch> [outdir]
# Required env: JT_PY (interpreter whose `import torch` is the Jittor shim) and
# RT_PY (independent real-PyTorch interpreter). Optional: RT_LIBSTDCXX, a
# libstdc++ to LD_PRELOAD for the real-torch side when its wheels need one.
set -uo pipefail
ARCH="${1:?usage: run_parity.sh <arch> [outdir]}"
OUT="${2:-${TMPDIR:-/tmp}/parity_${ARCH}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${JT_PY:?set JT_PY to the interpreter whose import torch resolves to the Jittor shim}"
: "${RT_PY:?set RT_PY to an independent real-PyTorch interpreter (the oracle)}"
RT_PRELOAD=()
if [ -n "${RT_LIBSTDCXX:-}" ]; then
    RT_PRELOAD=(LD_PRELOAD="$RT_LIBSTDCXX")
fi
NOISE='^\[i |^\[w |Compiling|cache_path|mpicc|addr2line|Total mem|Load cc|Writing model|Loading weights|Model config|DeprecationWarning|it/s\]'
COMMON_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8"

echo "### parity ${ARCH}  (out=${OUT})"
echo "--- JITTOR side (build + save + jt.grad) ---"
env $COMMON_ENV "$JT_PY" "$HERE/parity.py" jt "$ARCH" "$OUT" 2>&1 | grep -vE "$NOISE"
echo "--- REAL TORCH side (load same weights + backward) ---"
env $COMMON_ENV ${RT_PRELOAD[@]+"${RT_PRELOAD[@]}"} "$RT_PY" "$HERE/parity.py" rt "$ARCH" "$OUT" 2>&1 | grep -vE "$NOISE"
echo "--- COMPARE ---"
env $COMMON_ENV "$JT_PY" "$HERE/parity.py" cmp "$ARCH" "$OUT" 2>&1 | grep -vE "$NOISE"
