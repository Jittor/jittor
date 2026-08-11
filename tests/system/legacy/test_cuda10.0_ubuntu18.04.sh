cat > /tmp/cuda10.0-ubuntu18.04.dockerfile <<\EOF
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt update && apt install ca-certificates -y

RUN echo \
"deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" > /etc/apt/sources.list

# RUN rm -rf /var/lib/apt/lists/*
RUN apt update || true

RUN apt install wget \
        python3.7 python3.7-dev \
        g++ build-essential -y

WORKDIR /usr/src

RUN apt download python3-distutils && dpkg-deb -x ./python3-distutils* / \
    && wget -O - https://bootstrap.pypa.io/get-pip.py | python3.7

# change tsinghua mirror
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install jittor --timeout 100 && python3.7 -m jittor.selftest
RUN pip3 uninstall jittor -y

COPY . jittor
RUN python3.7 -m pip install ./jittor pytest==7.4.4
RUN python3.7 -m pytest jittor/tests/core/test_core.py -v
EOF

sudo docker build --tag jittor/jittor-cuda:10.0-18.04 -f /tmp/cuda10.0-ubuntu18.04.dockerfile .
sudo docker run --gpus all --rm jittor/jittor-cuda:10.0-18.04 bash -c \
"use_cuda=1 python3.7 -m jittor.selftest && \
python3.7 -m pytest jittor/tests/models/test_resnet.py -v && \
python3.7 -m pytest jittor/tests/compiler/test_parallel_pass.py -v && \
python3.7 -m pytest jittor/tests/compiler/test_atomic_tuner.py -v && \
python3.7 -m pytest jittor/tests/ops/test_where_op.py -v"
