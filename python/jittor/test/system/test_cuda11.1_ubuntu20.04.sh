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


RUN pip3 install jittor --timeout 100 && python3 -m jittor.test.test_example
RUN pip3 uninstall jittor -y

COPY . jittor
RUN python3 -m pip install ./jittor
RUN python3 -m jittor.test.test_core
EOF

sudo docker build --tag jittor/jittor-cuda:11.1-20.04 -f /tmp/cuda11.1-ubuntu20.04.dockerfile .
sudo docker run --gpus all --rm jittor/jittor-cuda:11.1-20.04 bash -c \
"python3 -m jittor.test.test_example && \
python3 -m jittor.test.test_resnet && \
python3 -m jittor.test.test_parallel_pass && \
python3 -m jittor.test.test_atomic_tuner && \
python3 -m jittor.test.test_where_op"