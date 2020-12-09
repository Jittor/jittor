cat > /tmp/cuda10.0-ubuntu16.04.dockerfile <<\EOF
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

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

RUN pip3 install jittor --timeout 100 && python3.7 -m jittor.test.test_example
RUN pip3 uninstall jittor -y

COPY . jittor
RUN python3.7 -m pip install ./jittor
RUN python3.7 -m jittor.test.test_core
EOF

sudo docker build --tag jittor/jittor-cuda:10.0-16.04 -f /tmp/cuda10.0-ubuntu16.04.dockerfile .
sudo docker run --gpus all --rm jittor/jittor-cuda:10.0-18.04 bash -c \
"python3.7 -m jittor.test.test_example && \
python3.7 -m jittor.test.test_resnet && \
python3.7 -m jittor.test.test_parallel_pass && \
python3.7 -m jittor.test.test_atomic_tuner && \
python3.7 -m jittor.test.test_where_op"