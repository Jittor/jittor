#!/usr/bin/python3
# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import os
import jittor_utils as jit_utils
home_path = jit_utils.home()

def run_cmd(cmd):
    print("RUN CMD:", cmd)
    assert os.system(cmd) == 0

# ubuntu
# debian9
# centos
# redhat8: docker run -it --name test registry.access.redhat.com/ubi8/ubi:8.1 bash
# fedora: docker run -it fedora
# archlinux: ???
# pacman fix: https://github.com/qutebrowser/qutebrowser/commit/478e4de7bd1f26bebdcdc166d5369b2b5142c3e2
# manjaro: ???

def run_in_centos(env):
    dockerfile_src = r"""
    FROM centos:7

    WORKDIR /root

    # install python
    RUN yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget -y
    RUN wget https://www.python.org/ftp/python/3.8.3/Python-3.8.3.tgz
    RUN tar xzf Python-3.8.3.tgz
    RUN yum install make -y
    RUN cd Python-3.8.3 && ./configure --enable-optimizations && make altinstall -j8

    # install g++-7
    # or yum install gcc-g++
    RUN yum install centos-release-scl -y
    RUN yum install devtoolset-7-gcc-c++ -y
    RUN yum install which -y
    RUN scl enable devtoolset-7 'g++ --version'
    RUN python3.8 -m pip install numpy tqdm pillow astunparse
    """

    with open("/tmp/centos_build_env", 'w') as f:
        f.write(dockerfile_src)


    centos_path = os.path.join(home_path, ".cache", "centos")
    os.makedirs(centos_path+"/src/jittor", exist_ok=True)
    os.makedirs(centos_path+"/src/jittor_utils", exist_ok=True)
    os.system(f"sudo cp -rL {jt.flags.jittor_path} {centos_path+'/src/'}")
    os.system(f"sudo cp -rL {jt.flags.jittor_path}/../jittor_utils {centos_path+'/src/'}")

    run_cmd(f"sudo docker build --tag centos_build_env -f /tmp/centos_build_env .")
    run_cmd(f"sudo docker run --rm -v {centos_path}:/root/.cache/jittor centos_build_env scl enable devtoolset-7 'PYTHONPATH=/root/.cache/jittor/src {env} python3.8 -m jittor.test.test_core'")
    run_cmd(f"sudo docker run --rm -v {centos_path}:/root/.cache/jittor centos_build_env scl enable devtoolset-7 'PYTHONPATH=/root/.cache/jittor/src {env} python3.8 -m jittor.test.test_core'")