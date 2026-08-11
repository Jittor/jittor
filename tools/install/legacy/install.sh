#!/usr/bin/env bash
set -euo pipefail

# Legacy provisioning helper. Prefer the maintained installation instructions
# in the repository README for new systems.
is_docker="${is_docker:-0}"
with_clang="${with_clang:-0}"
with_gcc="${with_gcc:-0}"
with_icc="${with_icc:-0}"
with_cuda="${with_cuda:-0}"
py_version="${py_version:-3.7}"
jittor_spec="${JITTOR_SPEC:-git+https://github.com/Jittor/jittor.git}"

if [[ ! "$py_version" =~ ^[0-9]+\.[0-9]+$ ]]; then
    printf 'py_version must look like 3.11, got %q\n' "$py_version" >&2
    exit 2
fi

selected_compilers=$((with_clang + with_gcc + with_icc))
if (( selected_compilers == 0 )); then
    with_gcc=1
elif (( selected_compilers > 1 )); then
    printf 'select exactly one of with_clang, with_gcc, or with_icc\n' >&2
    exit 2
fi

if [[ "$is_docker" == "1" ]]; then
    cat >/etc/apt/sources.list <<'EOF'
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
EOF
    rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
    apt-get update
    apt-get install --yes sudo lsb-release software-properties-common wget
fi

if [[ "$with_clang" == "1" ]]; then
    sudo apt-get update
    sudo apt-get install --yes clang libc++-dev libc++abi-dev libomp-dev
    export cc_path="${CC_PATH:-clang++}"
elif [[ "$with_gcc" == "1" ]]; then
    sudo apt-get update
    sudo apt-get install --yes g++ build-essential libomp-dev
    export cc_path="${CC_PATH:-g++}"
else
    if ! command -v icc >/dev/null 2>&1; then
        printf 'with_icc=1 was requested, but icc is unavailable\n' >&2
        exit 1
    fi
    export cc_path="${CC_PATH:-icc}"
fi

sudo add-apt-repository ppa:deadsnakes/ppa --yes
sudo apt-get update
sudo apt-get install --yes \
    "python${py_version}" \
    "python${py_version}-dev" \
    "python${py_version}-venv"

python_bin="python${py_version}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'requested Python interpreter is unavailable: %s\n' "$python_bin" >&2
    exit 1
fi

sudo "$python_bin" -m ensurepip --upgrade
sudo "$python_bin" -m pip install --upgrade "$jittor_spec"

if [[ "$with_cuda" == "1" ]]; then
    export nvcc_path="${NVCC_PATH:-/usr/local/cuda/bin/nvcc}"
fi

# Keep installed validation independent of the repository-only tests.
"$python_bin" -m jittor.selftest
if [[ "$with_cuda" == "1" ]]; then
    use_cuda=1 "$python_bin" -m jittor.selftest
fi

printf 'Jittor self-test passed. Export the environment values below if needed:\n'
printf 'export cc_path=%q\n' "$cc_path"
printf 'export nvcc_path=%q\n' "${nvcc_path:-}"
