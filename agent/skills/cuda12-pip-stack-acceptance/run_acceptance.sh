#!/usr/bin/env bash
# Clean-environment acceptance for `jittor[cuda12]` with a system CUDA visible.
#
# usage: run_acceptance.sh <uv|conda> <run-name> [system|unusable|old-nvcc|no-nvcc]
# env:   JITTOR_LAB_ROOT, DECOY_CUDA (a full CUDA toolkit root that poses as the
#        system CUDA), CUDA_VISIBLE_DEVICES (an idle card), optional
#        PYTHON_VERSION (default 3.11) and JITTOR_SOURCE (default: this checkout)
#
# Scenarios (see decoy_namespace.sh):
#   system    DECOY_CUDA's nvcc fits and works with the host compiler; Jittor
#             must compile with it
#   unusable  DECOY_CUDA's nvcc is in the version range but fails the host
#             compile check (e.g. CUDA 12.4 with gcc 14); only the check may
#             run it, Jittor must fall back to the pip nvcc
#   old-nvcc  the system nvcc reports 11.8; only --version may run it
#   no-nvcc   no system nvcc at all
# CUDA libraries must come from the pip wheels in every scenario.
set -euo pipefail
mode=${1:?uv or conda}
run=${2:?run name}
scenario=${3:-system}
case $scenario in
system) expect_nvcc=system decoy_mode=system ;;
unusable) expect_nvcc=pip decoy_mode=system ;;
old-nvcc|no-nvcc) expect_nvcc=pip decoy_mode=$scenario ;;
*) echo "scenario must be system, unusable, old-nvcc or no-nvcc" >&2; exit 2 ;;
esac
: "${JITTOR_LAB_ROOT:?set JITTOR_LAB_ROOT}"
: "${DECOY_CUDA:?set DECOY_CUDA to a CUDA toolkit root}"
: "${CUDA_VISIBLE_DEVICES:?select an idle GPU with CUDA_VISIBLE_DEVICES}"
here=$(cd "$(dirname "$0")" && pwd)
src=${JITTOR_SOURCE:-$(cd "$here/../../.." && pwd)}
py=${PYTHON_VERSION:-3.11}
topic=$JITTOR_LAB_ROOT/_state/cuda12-pip-stack
root=$topic/$run
[ ! -e "$root" ] || { echo "$root exists; use a new run name" >&2; exit 2; }
mkdir -p "$root"/{home,jittor-home,tmp,logs/ld,decoy-scratch}

export HOME=$root/home JITTOR_HOME=$root/jittor-home TMPDIR=$root/tmp
export UV_CACHE_DIR=$topic/uv-cache PIP_CACHE_DIR=$topic/pip-cache
export CONDA_PKGS_DIRS=$topic/conda-pkgs
unset nvcc_path JT_BUILD_NVCC_PATH CUDA_HOME CUDA_PATH PYTHONPATH VIRTUAL_ENV
env=$root/env
cd "$root"

case $mode in
uv)
    UV_PROJECT_ENVIRONMENT=$env uv sync --project "$src" --locked \
        --no-default-groups --extra cuda12 --python "$py" ;;
conda)
    conda create -y -q -p "$env" --override-channels -c conda-forge "python=$py" pip
    "$env/bin/python" -m pip install "$src[cuda12]" ;;
*)
    echo "mode must be uv or conda" >&2; exit 2 ;;
esac 2>&1 | tee "$root/logs/install.log"
"$env/bin/python" -m pip list 2>/dev/null | grep -iE '^(jittor|nvidia|cuda)' \
    > "$root/logs/cuda-packages.txt" || true

log=$root/logs/decoy-invocations.log
set +e
DECOY_CUDA=$DECOY_CUDA DECOY_LOG=$log DECOY_SCRATCH=$root/decoy-scratch \
DECOY_MODE=$decoy_mode CUDA12_PIP_EXPECT_NVCC=$expect_nvcc \
    unshare --user --map-root-user --mount \
    "$here/decoy_namespace.sh" \
    env LD_DEBUG=files LD_DEBUG_OUTPUT="$root/logs/ld/trace" \
    "$env/bin/python" "$here/gpu_smoke.py" \
    2>&1 | tee "$root/logs/smoke.log"
status=${PIPESTATUS[0]}
set -e
[ "$status" = 0 ] || { echo "FAIL: smoke exited $status" >&2; exit "$status"; }
grep -q '^CUDA12_PIP_SMOKE=' "$root/logs/smoke.log"

compiles=$(grep -v -- '--version' "$log" | grep -vc '/check\.cu' || true)
case $scenario in
system)
    [ "$compiles" -gt 0 ] || { echo "FAIL: the system nvcc compiled nothing" >&2; exit 1; } ;;
unusable)
    grep -q '/check\.cu' "$log" || { echo "FAIL: the host compile check never ran" >&2; exit 1; }
    [ "$compiles" = 0 ] || { echo "FAIL: the unusable system nvcc compiled:" >&2; grep -v -- '--version' "$log" | grep -v '/check\.cu' >&2; exit 1; } ;;
old-nvcc)
    [ "$compiles" = 0 ] || { echo "FAIL: the too-old system nvcc compiled:" >&2; grep -v -- '--version' "$log" >&2; exit 1; } ;;
no-nvcc)
    [ ! -s "$log" ] || { echo "FAIL: a hidden system nvcc ran:" >&2; cat "$log" >&2; exit 1; } ;;
esac

# Every process of the run, compiler subprocesses included. The system nvcc
# may run from the system toolkit; no CUDA runtime library may.
libs='lib(cudart|cudnn|cublas|nvrtc|cufft|curand|cusparse|nvJitLink|nccl)'
if grep -h "calling init: " "$root"/logs/ld/trace.* | grep -E \
    "calling init: (/usr/local/cuda|$DECOY_CUDA|/usr/lib/x86_64-linux-gnu)[^ ]*/$libs" \
    > "$root/logs/decoy-libraries.log"; then
    echo "FAIL: a process loaded CUDA libraries from the system CUDA:" >&2
    sort -u "$root/logs/decoy-libraries.log" >&2
    exit 1
fi
echo "system nvcc compile invocations: $compiles"
echo "ld traces: $(ls "$root"/logs/ld | wc -l) processes, no system CUDA library loaded"
echo "PASS: $mode $run $scenario ($root)"
