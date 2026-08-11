cat > /tmp/cuda11.1-ubuntu20.04.dockerfile <<\EOF
FROM nvidia/cuda:11.1-devel-ubuntu20.04

RUN apt update && apt install ca-certificates -y

RUN echo \
"deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" > /etc/apt/sources.list

# RUN rm -rf /var/lib/apt/lists/*
RUN apt update || true
RUN apt install g++ build-essential libomp-dev python3-dev python3-pip wget -y
RUN python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /usr/src/

RUN wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.0.5.39-1+cuda11.1_amd64.deb && \
    wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8-dev_8.0.5.39-1+cuda11.1_amd64.deb && \
    dpkg -i ./libcudnn8_8.0.5.39-1+cuda11.1_amd64.deb ./libcudnn8-dev_8.0.5.39-1+cuda11.1_amd64.deb && \
    rm *.deb
RUN ls


RUN pip3 install jittor --timeout 100 && python3 -m jittor.selftest
RUN pip3 uninstall jittor -y

COPY . jittor
RUN python3 -m pip install ./jittor pytest==7.4.4
RUN python3 -m pytest jittor/tests/core/test_core.py -v
EOF

sudo docker build --tag jittor/jittor-cuda:11.1-20.04 -f /tmp/cuda11.1-ubuntu20.04.dockerfile .
sudo docker run --gpus all --rm jittor/jittor-cuda:11.1-20.04 bash -c \
"use_cuda=1 python3 -m jittor.selftest && \
python3 -m pytest jittor/tests/models/test_resnet.py -v && \
python3 -m pytest jittor/tests/compiler/test_parallel_pass.py -v && \
python3 -m pytest jittor/tests/compiler/test_atomic_tuner.py -v && \
python3 -m pytest jittor/tests/ops/test_where_op.py -v"
