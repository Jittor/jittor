#!/usr/bin/env bash
# Clean-environment acceptance for `jittor[cuda12]` with a decoy system CUDA.
#
# usage: run_acceptance.sh <uv|conda> <run-name>
# env:   JITTOR_LAB_ROOT, DECOY_CUDA (a full CUDA toolkit root that poses as the
#        system CUDA), CUDA_VISIBLE_DEVICES (an idle card), optional
#        PYTHON_VERSION (default 3.11) and JITTOR_SOURCE (default: this checkout)
set -euo pipefail
mode=${1:?uv or conda}
run=${2:?run name}
: "${JITTOR_LAB_ROOT:?set JITTOR_LAB_ROOT}"
: "${DECOY_CUDA:?set DECOY_CUDA to a CUDA toolkit root}"
: "${CUDA_VISIBLE_DEVICES:?select an idle GPU with CUDA_VISIBLE_DEVICES}"
here=$(cd "$(dirname "$0")" && pwd)
src=${JITTOR_SOURCE:-$(cd "$here/../../.." && pwd)}
py=${PYTHON_VERSION:-3.11}
topic=$JITTOR_LAB_ROOT/_state/cuda12-pip-stack
root=$topic/$run
[ ! -e "$root" ] || { echo "$root exists; use a new run name" >&2; exit 2; }
mkdir -p "$root"/{home,jittor-home,tmp,logs,decoy-scratch}

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

mkdir -p "$root/logs/ld"
set +e
DECOY_CUDA=$DECOY_CUDA DECOY_LOG=$root/logs/decoy-invocations.log \
DECOY_SCRATCH=$root/decoy-scratch \
    unshare --user --map-root-user --mount \
    "$here/decoy_namespace.sh" \
    env LD_DEBUG=files LD_DEBUG_OUTPUT="$root/logs/ld/trace" \
    "$env/bin/python" "$here/gpu_smoke.py" \
    2>&1 | tee "$root/logs/smoke.log"
status=${PIPESTATUS[0]}
set -e
if [ -s "$root/logs/decoy-invocations.log" ]; then
    echo "FAIL: the decoy system toolchain was invoked:" >&2
    cat "$root/logs/decoy-invocations.log" >&2
    exit 1
fi
# Every process of the run, compiler subprocesses included, not just the
# smoke's own memory map.
if grep -h "calling init: " "$root"/logs/ld/trace.* | grep -E \
    "/usr/local/cuda|$DECOY_CUDA|/usr/lib/x86_64-linux-gnu/lib(cud|cublas|nvrtc)" \
    > "$root/logs/decoy-libraries.log"; then
    echo "FAIL: a process loaded CUDA libraries from the decoy:" >&2
    sort -u "$root/logs/decoy-libraries.log" >&2
    exit 1
fi
echo "ld traces: $(ls "$root"/logs/ld | wc -l) processes, no decoy library loaded"
[ "$status" = 0 ] || { echo "FAIL: smoke exited $status" >&2; exit "$status"; }
grep -q '^CUDA12_PIP_SMOKE=' "$root/logs/smoke.log"
echo "PASS: $mode $run ($root)"
