#!/usr/bin/python3
# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

# Polish steps:
# 1. create jittor-polish repo
# 2. copy jittor src into it
# 3. remove files
# 4. commit jittor-polish(check modify and break)
# 5. compile to build/$git_version/$cc_type/$use_cuda/a.obj
# 6. rsync to build-server
# 7. push to github
# 8. push to pip

import os
import jittor as jt
from jittor import LOG
from jittor.compiler import run_cmd
from jittor_utils import translator
import sys

jittor_path = os.path.realpath(os.path.join(jt.flags.jittor_path, "..", ".."))

polish_path = os.path.join(jittor_path, "..", "jittor-polish")
polish_path = os.path.realpath(polish_path)
build_path = polish_path + "/build"
LOG.i("Polish path:", polish_path)
if not os.path.isdir(polish_path):
    # create jittor-polish repo
    os.mkdir(polish_path)
    os.mkdir(build_path)
    run_cmd("git init . && git remote add origin git@github.com:Jittor/Jittor.git", polish_path)

# copy jittor src into it
names = "extern notebook python script src README.md README.src.md README.cn.md LICENSE.txt setup.py .gitignore".split()
for name in names:
    run_cmd(f"rsync -a {jittor_path}/{name} {polish_path}/")

git_version = run_cmd("git rev-parse HEAD", jittor_path)
LOG.i("git_version", git_version)
run_cmd(f"git rev-parse HEAD > {polish_path}/python/jittor/version", jittor_path)

# remove files
files = jt.compiler.files
file_to_delete = [ name for name in files
    if name.startswith("src") and \
        len(name.split("/"))==2 and name.endswith("node.cc")
]
LOG.i("file_to_delete", file_to_delete)
run_cmd(f"rm {' '.join(file_to_delete)}", polish_path)

# commit jittor-polish
run_cmd(f"git add .", polish_path)
status = run_cmd(f"git status", polish_path)
if "new file" not in status:
    LOG.i("Nothing change, exit...")
else:
    run_cmd(f"git commit -a -m 'version {git_version}'", polish_path)

# compile delete files
from pathlib import Path
home = str(Path.home())
for cc_type in ["g++", "clang"]:
    for device in ["cpu", "cuda"]:
        key = f"{git_version}-{cc_type}-{device}"
        env = f"cache_name=build/{cc_type}/{device} cc_path="
        cname = "g++" if cc_type=="g++" else "clang-8"
        env += cname
        # use core2 arch, avoid using avx instructions
        # TODO: support more archs, such as arm, or use ir(GIMPLE or LLVM)
        env += " cc_flags='-march=core2' "
        if device == "cpu":
            env += "nvcc_path='' "
        elif jt.flags.nvcc_path == "":
            env = "unset nvcc_path && " + env
        cmd = f"{env} {sys.executable} -c 'import jittor'"
        LOG.i("run cmd:", cmd)
        os.system(cmd)
        LOG.i("run cmd:", cmd)
        os.system(cmd)

        obj_path = home + f"/.cache/jittor/build/{cc_type}/{device}/{cname}/obj_files"
        obj_files = []
        for name in file_to_delete:
            name = name.split("/")[-1]
            fname = f"{obj_path}/{name}.o"
            assert os.path.isfile(fname), fname
            obj_files.append(fname)
        run_cmd(f"ld -r {' '.join(obj_files)} -o {build_path}/{key}.o")

# compress source
# tar -cvzf build/jittor.tgz . --exclude build --exclude .git --exclude .ipynb_checkpoints --exclude __pycache__
# mkdir -p jittor && tar -xvf ./jittor.tgz -C jittor
assert os.system(f"cd {polish_path} && tar --exclude=build --exclude=.git --exclude=.ipynb_checkpoints --exclude=__pycache__ -cvzf build/jittor.tgz . ")==0

# rsync to build-server
jittor_web_base_dir = "Documents/jittor-blog/assets/"
jittor_web_build_dir = jittor_web_base_dir + "build/"
assert os.system(f"rsync -avPu {polish_path}/build/ jittor-web:{jittor_web_build_dir}")==0
assert os.system(f"ssh jittor-web Documents/jittor-blog.git/hooks/post-update")==0

# push to github
# assert os.system(f"cd {polish_path} && git push -f origin master")==0

# push to pip