import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
JITTOR = REPO_ROOT / "python" / "jittor"


@pytest.fixture(scope="module")
def cache_compile_harness(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cache-compile-harness")
    source = directory / "main.cc"
    source.write_text(
        """
#include <string>
#include "utils/cache_compile.h"

int main(int argc, char** argv) {
    if (argc != 4) return 64;
    return jittor::jit_compiler::cache_compile(argv[1], argv[2], argv[3])
        ? 0 : 65;
}
""",
        encoding="utf-8",
    )
    executable = directory / "cache_compile_harness"
    subprocess.run(
        [
            os.environ.get("CXX", "g++"),
            "-std=c++14",
            "-I" + str(JITTOR / "src"),
            str(source),
            str(JITTOR / "src" / "utils" / "log.cc"),
            str(JITTOR / "src" / "utils" / "tracer.cc"),
            str(JITTOR / "src" / "utils" / "str_utils.cc"),
            str(JITTOR / "src" / "utils" / "cache_compile.cc"),
            "-lpthread",
            "-ldl",
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def test_dlink_products_and_keys_are_replaced_atomically(
        cache_compile_harness, tmp_path):
    """Readers must never observe a truncated wrapper product or cache key."""
    wrapper_name = "dlink_compiler.py"
    wrapper = tmp_path / wrapper_name
    wrapper.write_text(
        """#!/usr/bin/env python3
import sys

output = sys.argv[sys.argv.index("-o") + 1]
if "-MF" in sys.argv:
    dependency = sys.argv[sys.argv.index("-MF") + 1]
    with open(dependency, "w") as handle:
        handle.write(output + ": " + sys.argv[1] + "\\n")
with open(output, "wb") as handle:
    handle.write(b"complete shared library")
""",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    source = tmp_path / "kernel.cc"
    source.write_text("int kernel = 1;\n", encoding="utf-8")
    output = tmp_path / "kernel.so"
    output.write_bytes(b"old shared library")
    key = Path(str(output) + ".key")
    key.write_text("stale key\n", encoding="utf-8")
    old_output_inode = output.stat().st_ino
    old_key_inode = key.stat().st_ino
    (tmp_path / "obj_files").mkdir()

    command = '%s %s -o %s' % (wrapper, source, output)
    result = subprocess.run(
        [
            str(cache_compile_harness),
            command,
            str(tmp_path),
            str(JITTOR),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    assert output.read_bytes() == b"complete shared library"
    assert output.stat().st_ino != old_output_inode
    assert key.stat().st_ino != old_key_inode
    assert not list(tmp_path.glob("*.tmp.*"))


@pytest.fixture
def fake_compiler(tmp_path):
    compiler = tmp_path / "fake-compiler"
    compiler.write_text(
        """#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
output_index = args.index("-o")
output = args[output_index + 1]
inputs = [arg for index, arg in enumerate(args)
          if index != output_index + 1 and arg.endswith((".s", ".o"))]
if inputs and not all(os.path.isfile(path) for path in inputs):
    raise SystemExit("missing input: " + repr(inputs))
if "-MF" in args:
    dependency = args[args.index("-MF") + 1]
    sources = [arg for index, arg in enumerate(args)
               if index != output_index + 1
               and arg.endswith((".cc", ".cu", ".s", ".o"))]
    with open(dependency, "w") as handle:
        handle.write(output + ": " + " ".join(sources) + "\\n")
with open(output, "wb") as handle:
    if output.endswith(".post.s"):
        handle.write(b"\\t.text\\n")
    else:
        handle.write(b"complete product")
""",
        encoding="utf-8",
    )
    compiler.chmod(0o755)
    return compiler


def test_dlink_preserves_a_private_shared_library_output(
        tmp_path, fake_compiler):
    source = tmp_path / "kernel_op.cu"
    source.write_text("int kernel = 1;\n", encoding="utf-8")
    private_output = tmp_path / "kernel_op.so.tmp.123"
    dependency = tmp_path / "kernel.d.tmp.123"
    result = subprocess.run(
        [
            sys.executable,
            str(JITTOR / "utils" / "dlink_compiler.py"),
            str(fake_compiler),
            str(source),
            "-dc",
            "-MD",
            "-MF",
            str(dependency),
            "-o",
            str(private_output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "kernel_op.o").read_bytes() == b"complete product"
    assert private_output.read_bytes() == b"complete product"
    assert not (tmp_path / "kernel_op.so").exists()
    dep_inputs = dependency.read_text(encoding="utf-8").split(": ", 1)[1]
    assert "kernel_op.cu" in dep_inputs
    assert "kernel_op.o" not in dep_inputs


def _dependency_entries(key):
    entries = set()
    for line in key.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^# (.*): [0-9a-f]{64}$", line)
        if match:
            entries.add(Path(match.group(1)).name)
    return entries


def test_compiler_depfile_owns_the_cache_dependencies(
        cache_compile_harness, tmp_path):
    include = tmp_path / "include"
    include.mkdir()
    for name in ("quoted.h", "angled.h", "conditional.h", "macro.h",
                 "inactive.h"):
        (include / name).write_text("// %s\n" % name, encoding="utf-8")
    source = tmp_path / "kernel.cc"
    source.write_text(
        """
#include "quoted.h"
#include <angled.h>
#if defined(ENABLE_CONDITIONAL)
#include "conditional.h"
#endif
#define MACRO_HEADER "macro.h"
#include MACRO_HEADER
#if 0
#include "inactive.h"
#endif
extern "C" int kernel() { return 1; }
""",
        encoding="utf-8",
    )
    output = tmp_path / "kernel.so"
    (tmp_path / "obj_files").mkdir()
    command = "%s -shared -fPIC -DENABLE_CONDITIONAL -I%s %s -o %s" % (
        os.environ.get("CXX", "g++"), include, source, output)
    arguments = [
        str(cache_compile_harness), command, str(tmp_path), str(JITTOR)]

    first = subprocess.run(arguments, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    assert first.returncode == 0, first.stdout
    entries = _dependency_entries(Path(str(output) + ".key"))
    assert {"quoted.h", "angled.h", "conditional.h", "macro.h"} <= entries
    assert "inactive.h" not in entries
    assert Path(str(output) + ".d").is_file()

    cached = subprocess.run(arguments)
    assert cached.returncode == 65
    (include / "conditional.h").write_text(
        "// conditional.h changed\n", encoding="utf-8")
    rebuilt = subprocess.run(arguments, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    assert rebuilt.returncode == 0, rebuilt.stdout
