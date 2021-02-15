#!/usr/bin/python3
# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

# Publish steps:
# 1. build,push,upload docker image[jittor/jittor]
# 2. build,push,upload docker image[jittor/jittor-cuda]
# upload to pip:
# rm -rf dist && python3.7 ./setup.py sdist && python3.7 -m twine upload dist/*
import os

def run_cmd(cmd):
    print("[run cmd]", cmd)
    assert os.system(cmd) == 0

def upload_file(path):
    run_cmd(f"rsync -avPu {path} jittor-web:Documents/jittor-blog/assets/build/")

def docker_task(name, build_cmd):
    run_cmd(build_cmd)
    run_cmd(f"sudo docker push {name}")
    bname = os.path.basename(name)
    run_cmd(f"sudo docker save {name}:latest -o /tmp/{bname}.tgz && sudo chmod 666 /tmp/{bname}.tgz")
    upload_file(f"/tmp/{bname}.tgz")

docker_task(
    "jittor/jittor-cuda-11-1", 
    "sudo docker build --tag jittor/jittor-cuda-11-1:latest -f script/Dockerfile_cuda11 . --network host"
)

docker_task(
    "jittor/jittor", 
    "sudo docker build --tag jittor/jittor:latest . --network host"
)

docker_task(
    "jittor/jittor-cuda", 
    "sudo docker build --tag jittor/jittor-cuda:latest --build-arg FROM_IMAGE='nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04' . --network host"
)

docker_task(
    "jittor/jittor-cuda-10-1", 
    "sudo docker build --tag jittor/jittor-cuda-10-1:latest --build-arg FROM_IMAGE='nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04' . --network host"
)

run_cmd("ssh jittor-web Documents/jittor-blog.git/hooks/post-update")