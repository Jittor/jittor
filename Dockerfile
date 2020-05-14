FROM ubuntu:20.04

RUN apt update && apt install ca-certificates -y

# change tsinghua mirror
RUN echo \
"deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse\n\
deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" > /etc/apt/sources.list

RUN apt update && apt install wget \
        python3 python3-dev python3-pip \
        g++ build-essential -y

ENV PYTHONIOENCODING utf8

# change tsinghua mirror
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install  \
        pybind11 \
        numpy \
        tqdm \
        pillow \
        astunparse

WORKDIR /usr/src/jittor
COPY . .

RUN pip3 install . --timeout 100

RUN python3 -m jittor.test.test_example

RUN rm -rf ~/.cache/jittor/default