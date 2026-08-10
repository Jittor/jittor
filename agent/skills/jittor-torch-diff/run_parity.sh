#!/usr/bin/env bash
# Drive a full jittor<->real-torch forward+backward parity run across both envs.
# Usage: run_parity.sh <arch> [outdir]
# Env overrides: JT_PY, RT_PY, RT_LIBSTDCXX (LD_PRELOAD target for real torch).
set -uo pipefail
ARCH="${1:?usage: run_parity.sh <arch> [outdir]}"
OUT="${2:-${TMPDIR:-/tmp}/parity_${ARCH}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JT_PY="${JT_PY:-/home/yizhang/miniconda3/envs/jt-torch/bin/python}"
RT_PY="${RT_PY:-/home/yizhang/miniconda3/envs/rt/bin/python}"
RT_LIBSTDCXX="${RT_LIBSTDCXX:-/home/yizhang/miniconda3/envs/rt/lib/libstdc++.so.6}"
NOISE='^\[i |^\[w |Compiling|cache_path|mpicc|addr2line|Total mem|Load cc|Writing model|Loading weights|Model config|DeprecationWarning|it/s\]'
COMMON_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8"

echo "### parity ${ARCH}  (out=${OUT})"
echo "--- JITTOR side (build + save + jt.grad) ---"
env $COMMON_ENV "$JT_PY" "$HERE/parity.py" jt "$ARCH" "$OUT" 2>&1 | grep -vE "$NOISE"
echo "--- REAL TORCH side (load same weights + backward) ---"
env $COMMON_ENV LD_PRELOAD="$RT_LIBSTDCXX" "$RT_PY" "$HERE/parity.py" rt "$ARCH" "$OUT" 2>&1 | grep -vE "$NOISE"
echo "--- COMPARE ---"
env $COMMON_ENV "$JT_PY" "$HERE/parity.py" cmp "$ARCH" "$OUT" 2>&1 | grep -vE "$NOISE"
