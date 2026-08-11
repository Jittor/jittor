#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="${cache_path:-${1:-$REPO_ROOT/python/jittor/extern/mkl}}"
ARCHIVE="mkldnn_lnx_1.0.2_cpu_gomp.tgz"
DIRECTORY="mkldnn_lnx_1.0.2_cpu_gomp"
URL="https://github.com/intel/mkl-dnn/releases/download/v1.0.2/$ARCHIVE"

for tool in g++ tar wget; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'required installer tool is unavailable: %s\n' "$tool" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

if [[ ! -f "$ARCHIVE" ]]; then
    wget --output-document "$ARCHIVE" "$URL"
fi
if [[ ! -d "$DIRECTORY" ]]; then
    tar -xzf "$ARCHIVE"
fi

if [[ ! -x "$DIRECTORY/examples/test" ]]; then
    printf 'compiling and running the legacy oneDNN example\n'
    (
        cd "$DIRECTORY/examples"
        g++ -std=c++14 cpu_cnn_inference_f32.cpp -Ofast -lmkldnn \
            -I ../include -L ../lib -o test
        LD_LIBRARY_PATH="$(cd ../lib && pwd):${LD_LIBRARY_PATH:-}" ./test
    )
fi
