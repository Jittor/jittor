#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/jittor-lab}"

SOURCE_DIR="${1:-$PWD}"
OUTPUT_DIR="${2:-$LAB_ROOT/_state/tools/build-aarch64-mkl}"
BUILD_DIR="${JITTOR_BUILD_DIR:-$OUTPUT_DIR/build}"
JOBS="${JITTOR_BUILD_JOBS:-}"
CC_AARCH64="${CC_AARCH64:-aarch64-linux-gnu-gcc-8}"
CXX_AARCH64="${CXX_AARCH64:-aarch64-linux-gnu-g++-8}"
ARCHIVE_NAME="dnnl_lnx_2.2.0_cpu_gomp_aarch64"

if [[ -z "$JOBS" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS="$(nproc)"
    else
        JOBS=1
    fi
fi

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'JITTOR_BUILD_JOBS must be a positive integer, got %q\n' "$JOBS" >&2
    exit 2
fi

for tool in cmake tar sha256sum "$CC_AARCH64" "$CXX_AARCH64"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'required build tool is unavailable: %s\n' "$tool" >&2
        exit 1
    fi
done

if [[ ! -f "$SOURCE_DIR/CMakeLists.txt" ]]; then
    printf 'oneDNN source directory is invalid: %s\n' "$SOURCE_DIR" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$BUILD_DIR"

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=AARCH64 \
    -DCMAKE_C_COMPILER="$CC_AARCH64" \
    -DCMAKE_CXX_COMPILER="$CXX_AARCH64" \
    -DCMAKE_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel "$JOBS"

PACKAGE_DIR="$BUILD_DIR/$ARCHIVE_NAME"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/lib" "$PACKAGE_DIR/include/oneapi/dnnl"
cp -R "$SOURCE_DIR/include/." "$PACKAGE_DIR/include/"
cp "$BUILD_DIR/src/libmkldnn.so" "$PACKAGE_DIR/lib/libmkldnn.so"
cp -R "$SOURCE_DIR/examples" "$PACKAGE_DIR/examples"
cp -R "$BUILD_DIR/include/oneapi/dnnl/." "$PACKAGE_DIR/include/oneapi/dnnl/"

tar -C "$BUILD_DIR" -czf "$OUTPUT_DIR/$ARCHIVE_NAME.tgz" "$ARCHIVE_NAME"
printf 'local artifact: %s\n' "$OUTPUT_DIR/$ARCHIVE_NAME.tgz"
sha256sum "$OUTPUT_DIR/$ARCHIVE_NAME.tgz"
