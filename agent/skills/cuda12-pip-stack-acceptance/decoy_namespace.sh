#!/usr/bin/env bash
# Run a command with a real CUDA toolkit posing as the system CUDA.
#
# Must run as root inside `unshare --user --map-root-user --mount`; nothing
# outside the private mount namespace changes. Inside, $DECOY_CUDA appears as
# /usr/local/cuda, /usr/bin/nvcc points at it, its libcudart/libcudnn/libcublas/
# libnvrtc sit in /usr/lib/x86_64-linux-gnu, and PATH, LD_LIBRARY_PATH,
# CUDA_HOME and CUDA_PATH name it. Its nvcc and ptxas are wrapped to append
# every invocation to $DECOY_LOG, so any use is recorded even if it succeeds.
#
# usage: decoy_namespace.sh <command> [args...]
# env:   DECOY_CUDA (toolkit root), DECOY_LOG (file), DECOY_SCRATCH (empty dir)
set -euo pipefail
: "${DECOY_CUDA:?toolkit root to pose as system CUDA}"
: "${DECOY_LOG:?file that records decoy compiler invocations}"
: "${DECOY_SCRATCH:?empty scratch directory}"
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
    printf '#!/bin/sh\necho "%s $*" >> "%s"\nexec "%s" "$@"\n' \
        "$tool" "$DECOY_LOG" "$DECOY_CUDA/bin/$tool" > "$wrapper"
    chmod +x "$wrapper"
    mount --bind "$wrapper" "/usr/local/cuda/bin/$tool"
done

overlay /usr/bin usr-bin
ln -sf /usr/local/cuda/bin/nvcc /usr/bin/nvcc

libdir=/usr/lib/x86_64-linux-gnu
overlay "$libdir" usr-lib
for lib in "$DECOY_CUDA"/lib64/lib{cudart,cudnn,cublas,cublasLt,nvrtc}*.so*; do
    [ -e "$lib" ] && ln -sf "/usr/local/cuda/lib64/$(basename "$lib")" "$libdir/"
done

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export CUDA12_PIP_FORBIDDEN="/usr/local/cuda:$DECOY_CUDA:$libdir"

[ "$(command -v nvcc)" = /usr/local/cuda/bin/nvcc ] || { echo "decoy nvcc is not first on PATH" >&2; exit 2; }
"$@"
