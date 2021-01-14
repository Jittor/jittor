#!/usr/bin/python3
# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from ctypes import cdll
import sys

lib_path = sys.argv[1]
cmd = sys.argv[2]
if not lib_path.endswith(".so"):
    i = -1
    while lib_path[i] != '.':
        i -= 1
    if i > -10: lib_path = lib_path[:i]
    lib_path += ".so"

if cmd == "run_so":
    lib = cdll.LoadLibrary(lib_path)
    lib.fake_main()
    exit(0)

with open(lib_path+".key") as f:
    cpcmd = f.read().splitlines()[0]

def run_cmd(cmd):
    print("Run cmd:", cmd)
    assert os.system(cmd) == 0, "Run cmd failed: "+cmd

import os
if cmd == "cc_to_so":
    run_cmd(cpcmd)
    # remove hash info, force re-compile
    with open(lib_path+'.key', 'w') as f:
        f.write(cpcmd)
elif cmd == "cc_to_s":
    asm_cmd = cpcmd.replace("_op.so", "_op.s") \
        .replace("-g", "") \
        .replace("-lstdc++", "") \
        .replace("-ldl", "") \
        .replace("-shared", "-S")
    run_cmd(asm_cmd)
elif cmd == "s_to_so":
    asm_cmd = cpcmd.replace("_op.cc", "_op.s") \
        .replace("-g", "")
    run_cmd(asm_cmd)
    # remove hash info, force re-compile
    with open(lib_path+'.key', 'w') as f:
        f.write(cpcmd)
elif cmd == "perf_so":
    perf_cmd = "perf record "+__file__+" "+lib_path+" run_so && perf annotate"
    run_cmd(perf_cmd)
elif cmd == "vtune_so":
    if os.path.isdir("./__res"):
        run_cmd("rm -r ./__res")
    vtune_cmd = "amplxe-cl -collect uarch-exploration -r ./__res "+__file__+" "+lib_path+" run_so"
    run_cmd(vtune_cmd)
else:
    assert 0, "unknown cmd: {cmd}".format(cmd)
