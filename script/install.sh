#!/bin/bash
# Single line install script
# wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install.sh | with_clang=1 with_cuda=1 bash
set -ex

if [ "$is_docker" = "1" ]; then
tee /etc/apt/sources.list <<EOF
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
EOF
rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list
apt update
apt install sudo lsb-release software-properties-common wget -y
fi

# Step 1: Choose your back-end compiler

if [ "$with_clang" = "1" ]; then
sudo apt install wget lsb-release software-properties-common -y
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install_llvm.sh > /tmp/llvm.sh
sudo bash /tmp/llvm.sh 8
sudo apt-get install libc++-8-dev libc++abi-8-dev -y
sudo apt-get install libomp-8-dev -y
export cc_path="clang-8"
fi

if [ "$with_gcc" = "1" ]; then
sudo apt install g++ build-essential libomp-dev -y
export cc_path="g++"
fi

if [ "$with_icc" = "1" ]; then
export cc_path="icc"
fi

# Step 2: Install Python and dependency

if [ "$py_version" = "" ]; then
py_version="3.7"
fi
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt install python$py_version python$py_version-dev -y
# python3.8 need this
# sudo apt install python3.8-distutils
wget -O - https://bootstrap.pypa.io/get-pip.py | sudo -H python$py_version

# Step 3: Run jittor

sudo python$py_version -m pip install git+https://github.com/Jittor/jittor.git

if [ "$with_cuda" = "1" ]; then
export nvcc_path="/usr/local/cuda/bin/nvcc"
fi

# run a simple test
python$py_version -m jittor.test.test_example

if [ "$with_cuda" = "1" ]; then
python$py_version -m jittor.test.test_cuda
fi

set +x
echo "jittor test is passed. Please export the following enviroments value"
echo "---------------------------------------"
echo "export cc_path=$cc_path"
echo "export nvcc_path=$nvcc_path"
echo "---------------------------------------"