#!/usr/bin/env python3
"""Measure the cold cost of one generated C++ kernel.

The phases are intentionally explicit and independent:
source generation, Jittor cache_compile, compiler, linker, and dynamic load.
This script is a diagnostic harness; it does not modify Jittor's compiler or
optimization settings.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


SOURCE = r'''#include <cstddef>
extern "C" int jittor_profile_kernel(const float* input, float* output,
                                      std::size_t size) {
    for (std::size_t i = 0; i < size; ++i)
        output[i] = input[i] * 2.0f + 1.0f;
    return size != 0;
}
'''


def clocked(function):
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def run(command, *, cwd):
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            "command failed ({}): {}\n{}".format(
                completed.returncode, shlex.join(command), completed.stdout
            )
        )
    return elapsed, completed.stdout


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--json", type=Path, help="write the JSON report here")
    parser.add_argument("--work-dir", type=Path, help="keep generated files here")
    parser.add_argument(
        "--no-cache-compile",
        action="store_true",
        help="skip the Jittor cache_compile phase",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    compiler = shutil.which(args.compiler) or args.compiler
    if not shutil.which(compiler) and not Path(compiler).exists():
        raise SystemExit("compiler not found: {}".format(args.compiler))

    temporary = args.work_dir is None
    root = Path(tempfile.mkdtemp(prefix="jittor-single-kernel-")) if temporary else args.work_dir
    root.mkdir(parents=True, exist_ok=True)
    source = root / "profile_kernel.cc"
    cache_root = root / "cache"
    cache_root.mkdir()
    cache_output = root / "cache_kernel.so"
    object_file = root / "direct_kernel.o"
    shared_object = root / "direct_kernel.so"
    phases = {}
    commands = {}
    try:
        _, generation_time = clocked(lambda: source.write_text(SOURCE, encoding="utf-8"))
        phases["source_generation"] = generation_time

        cache_command = [
            compiler,
            "-std=c++14",
            "-O0",
            "-fPIC",
            "-shared",
            str(source),
            "-o",
            str(cache_output),
        ]
        commands["cache_compile"] = cache_command
        cache_available = False
        if not args.no_cache_compile:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
                import jittor_utils

                jittor_utils.try_import_jit_utils_core(True)
                cache_available = jittor_utils.cc is not None
                started = time.perf_counter()
                if cache_available:
                    jittor_utils.cc.cache_compile(
                        shlex.join(cache_command), str(cache_root), str(source.parent)
                    )
                else:
                    jittor_utils.run_cmd(shlex.join(cache_command))
                phases["cache_compile"] = time.perf_counter() - started
            except Exception:
                raise

        compile_command = [
            compiler,
            "-std=c++14",
            "-O0",
            "-fPIC",
            "-c",
            str(source),
            "-o",
            str(object_file),
        ]
        commands["compiler"] = compile_command
        phases["compiler"], _ = run(compile_command, cwd=root)

        link_command = [compiler, "-shared", str(object_file), "-o", str(shared_object)]
        commands["link"] = link_command
        phases["link"], _ = run(link_command, cwd=root)

        def load_library():
            library = ctypes.CDLL(str(shared_object))
            library.jittor_profile_kernel.restype = ctypes.c_int
            return library

        library, phases["load"] = clocked(load_library)
        result = library.jittor_profile_kernel(
            (ctypes.c_float * 1)(2.0), (ctypes.c_float * 1)(), 1
        )
        if result != 1:
            raise RuntimeError("loaded kernel returned {}".format(result))

        report = {
            "schema": 1,
            "compiler": compiler,
            "source_bytes": len(SOURCE.encode("utf-8")),
            "phases_seconds": phases,
            "commands": {name: shlex.join(command) for name, command in commands.items()},
            "cache_compile": {
                "requested": not args.no_cache_compile,
                "available": cache_available,
                "native_binding": "jit_utils_core" if cache_available else "python-fallback",
            },
            "work_dir": str(root),
        }
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if temporary:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
