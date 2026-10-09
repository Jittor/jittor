#!/usr/bin/env bash
# Run a command with a real CUDA toolkit posing as the system CUDA.
#
# Must run as root inside `unshare --user --map-root-user --mount`; nothing
# outside the private mount namespace changes. Inside, $DECOY_CUDA appears as
# /usr/local/cuda, its libcudart/libcudnn/libcublas/libnvrtc sit in
# /usr/lib/x86_64-linux-gnu, and LD_LIBRARY_PATH, CUDA_HOME and CUDA_PATH name
# it. Its nvcc and ptxas are wrapped to append every invocation to $DECOY_LOG.
#
# DECOY_MODE picks what the system compiler looks like:
#   system    nvcc first on PATH and at /usr/bin/nvcc (the real one, 12.4)
#   old-nvcc  same, but `nvcc --version` reports 11.8, too old for the stack
#   no-nvcc   libraries only: /usr/local/cuda/bin is empty, nothing on PATH
#
# usage: decoy_namespace.sh <command> [args...]
# env:   DECOY_CUDA (toolkit root), DECOY_LOG (file), DECOY_SCRATCH (empty dir),
#        DECOY_MODE (default system)
set -euo pipefail
: "${DECOY_CUDA:?toolkit root to pose as system CUDA}"
: "${DECOY_LOG:?file that records decoy compiler invocations}"
: "${DECOY_SCRATCH:?empty scratch directory}"
mode=${DECOY_MODE:-system}
case $mode in system|old-nvcc|no-nvcc) ;; *) echo "bad DECOY_MODE $mode" >&2; exit 2 ;; esac
[ "$(id -u)" = 0 ] || { echo "run inside unshare --user --map-root-user --mount" >&2; exit 2; }
[ -x "$DECOY_CUDA/bin/nvcc" ] || { echo "$DECOY_CUDA/bin/nvcc missing" >&2; exit 2; }

mount -t tmpfs tmpfs "$DECOY_SCRATCH"
overlay() {
    local target=$1 name=$2
    mkdir -p "$DECOY_SCRATCH/$name/upper" "$DECOY_SCRATCH/$name/work"
    mount -t overlay overlay \
        -o "lowerdir=$target,upperdir=$DECOY_SCRATCH/$name/upper,workdir=$DECOY_SCRATCH/$name/work" \
        "$target"
}

overlay /usr/local usr-local
mkdir -p /usr/local/cuda
mount --bind "$DECOY_CUDA" /usr/local/cuda
: > "$DECOY_LOG"
for tool in nvcc ptxas; do
    wrapper="$DECOY_SCRATCH/$tool"
    {
        echo '#!/bin/sh'
        echo "echo \"$tool \$*\" >> \"$DECOY_LOG\""
        if [ "$tool" = nvcc ] && [ "$mode" = old-nvcc ]; then
            echo 'if [ "$1" = --version ]; then'
            echo '  echo "Cuda compilation tools, release 11.8, V11.8.89"; exit 0'
            echo 'fi'
        fi
        echo "exec \"$DECOY_CUDA/bin/$tool\" \"\$@\""
    } > "$wrapper"
    chmod +x "$wrapper"
    mount --bind "$wrapper" "/usr/local/cuda/bin/$tool"
done

if [ "$mode" = no-nvcc ]; then
    mount -t tmpfs tmpfs /usr/local/cuda/bin
else
    overlay /usr/bin usr-bin
    ln -sf /usr/local/cuda/bin/nvcc /usr/bin/nvcc
    export PATH="/usr/local/cuda/bin:$PATH"
fi

libdir=/usr/lib/x86_64-linux-gnu
overlay "$libdir" usr-lib
for lib in "$DECOY_CUDA"/lib64/lib{cudart,cudnn,cublas,cublasLt,nvrtc}*.so*; do
    [ -e "$lib" ] && ln -sf "/usr/local/cuda/lib64/$(basename "$lib")" "$libdir/"
done

export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export CUDA12_PIP_FORBIDDEN="/usr/local/cuda:$DECOY_CUDA:$libdir"

if [ "$mode" = no-nvcc ]; then
    ! command -v nvcc >/dev/null || { echo "an nvcc is still on PATH" >&2; exit 2; }
else
    [ "$(command -v nvcc)" = /usr/local/cuda/bin/nvcc ] || { echo "decoy nvcc is not first on PATH" >&2; exit 2; }
fi
"$@"
