#!/usr/bin/python3
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from ctypes import cdll
import os
import shlex
import sys


def _shell_join(arguments):
    return " ".join(shlex.quote(argument) for argument in arguments)


def rewrite_compile_command(command, mode):
    arguments = shlex.split(command)
    if mode == "cc_to_s":
        remove = {"-g", "-lstdc++", "-ldl"}
        arguments = [argument for argument in arguments if argument not in remove]
        arguments = [
            "-S" if argument == "-shared" else argument.replace("_op.so", "_op.s")
            for argument in arguments
        ]
    elif mode == "s_to_so":
        arguments = [argument for argument in arguments if argument != "-g"]
        arguments = [argument.replace("_op.cc", "_op.s") for argument in arguments]
    else:
        raise ValueError("unsupported compile rewrite: " + mode)
    return _shell_join(arguments)


def run_cmd(command):
    print("Run cmd:", command)
    assert os.system(command) == 0, "Run cmd failed: " + command


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    lib_path = argv[0]
    cmd = argv[1]
    if not lib_path.endswith(".so"):
        i = -1
        while lib_path[i] != '.':
            i -= 1
        if i > -10:
            lib_path = lib_path[:i]
        lib_path += ".so"

    if cmd == "run_so":
        lib = cdll.LoadLibrary(lib_path)
        lib.fake_main()
        return 0

    with open(lib_path + ".key") as f:
        cpcmd = f.read().splitlines()[0]

    if cmd == "cc_to_so":
        run_cmd(cpcmd)
        # Remove hash info and force re-compilation.
        with open(lib_path + '.key', 'w') as f:
            f.write(cpcmd)
    elif cmd == "cc_to_s":
        run_cmd(rewrite_compile_command(cpcmd, cmd))
    elif cmd == "s_to_so":
        run_cmd(rewrite_compile_command(cpcmd, cmd))
        # Remove hash info and force re-compilation.
        with open(lib_path + '.key', 'w') as f:
            f.write(cpcmd)
    elif cmd == "perf_so":
        perf_cmd = "perf record " + __file__ + " " + lib_path + " run_so && perf annotate"
        run_cmd(perf_cmd)
    elif cmd == "vtune_so":
        if os.path.isdir("./__res"):
            run_cmd("rm -r ./__res")
        vtune_cmd = "amplxe-cl -collect uarch-exploration -r ./__res " \
            + __file__ + " " + lib_path + " run_so"
        run_cmd(vtune_cmd)
    else:
        raise AssertionError("unknown cmd: {cmd}".format(cmd=cmd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
