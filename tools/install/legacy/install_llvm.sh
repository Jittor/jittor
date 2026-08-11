#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Legacy LLVM 8-10 installer for obsolete Debian/Ubuntu releases.

Usage:
  sudo install_llvm.sh --acknowledge-legacy [8|9|10]

Use the distribution's supported clang/libomp packages on maintained systems.
EOF
}

if [[ "${1:-}" != "--acknowledge-legacy" ]]; then
    usage >&2
    exit 2
fi
shift

LLVM_VERSION="${1:-8}"
case "$LLVM_VERSION" in
    8) LLVM_VERSION_STRING="-8" ;;
    9) LLVM_VERSION_STRING="-9" ;;
    10) LLVM_VERSION_STRING="" ;;
    *)
        printf 'unsupported legacy LLVM version: %s\n' "$LLVM_VERSION" >&2
        exit 3
        ;;
esac

if [[ "$EUID" -ne 0 ]]; then
    printf 'this legacy installer must run as root\n' >&2
    exit 1
fi

for tool in add-apt-repository apt-get apt-key lsb_release wget; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'required installer tool is unavailable: %s\n' "$tool" >&2
        exit 1
    fi
done

DISTRO="$(lsb_release -is)"
VERSION="$(lsb_release -sr)"
DIST_VERSION="${DISTRO}_${VERSION}"

case "$DIST_VERSION" in
    Debian_9*) REPO_NAME="deb http://apt.llvm.org/stretch/ llvm-toolchain-stretch${LLVM_VERSION_STRING} main" ;;
    Debian_10*) REPO_NAME="deb http://apt.llvm.org/buster/ llvm-toolchain-buster${LLVM_VERSION_STRING} main" ;;
    Debian_unstable|Debian_testing) REPO_NAME="deb http://apt.llvm.org/unstable/ llvm-toolchain${LLVM_VERSION_STRING} main" ;;
    Ubuntu_16.04*) REPO_NAME="deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial${LLVM_VERSION_STRING} main" ;;
    Ubuntu_18.04*) REPO_NAME="deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic${LLVM_VERSION_STRING} main" ;;
    Ubuntu_18.10*) REPO_NAME="deb http://apt.llvm.org/cosmic/ llvm-toolchain-cosmic${LLVM_VERSION_STRING} main" ;;
    Ubuntu_19.04*) REPO_NAME="deb http://apt.llvm.org/disco/ llvm-toolchain-disco${LLVM_VERSION_STRING} main" ;;
    *)
        printf 'unsupported legacy distribution: %s\n' "$DIST_VERSION" >&2
        exit 2
        ;;
esac

# apt-key is intentionally retained only for these end-of-life distributions.
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
add-apt-repository "$REPO_NAME"
apt-get update
apt-get install --yes \
    "clang-${LLVM_VERSION}" \
    "lldb-${LLVM_VERSION}" \
    "lld-${LLVM_VERSION}" \
    "clangd-${LLVM_VERSION}"
