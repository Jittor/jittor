import os
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


@pytest.mark.parametrize("wrapper_name", ["asm_tuner.py", "dlink_compiler.py"])
def test_wrapped_products_and_keys_are_replaced_atomically(
        cache_compile_harness, tmp_path, wrapper_name):
    """Readers must never observe a truncated wrapper product or cache key."""
    wrapper = tmp_path / wrapper_name
    wrapper.write_text(
        """#!/usr/bin/env python3
import sys

output = sys.argv[sys.argv.index("-o") + 1]
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


def test_asm_tuner_preserves_a_private_shared_library_output(
        tmp_path, fake_compiler):
    source = tmp_path / "kernel_op.cc"
    source.write_text("int kernel = 1;\n", encoding="utf-8")
    private_output = tmp_path / "kernel_op.so.tmp.123"
    result = subprocess.run(
        [
            sys.executable,
            str(JITTOR / "utils" / "asm_tuner.py"),
            "--cc_path=" + str(fake_compiler),
            str(source),
            "-shared",
            "-o",
            str(private_output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    assert private_output.read_bytes() == b"complete product"
    assert not (tmp_path / "kernel_op.so").exists()


def test_dlink_preserves_a_private_shared_library_output(
        tmp_path, fake_compiler):
    source = tmp_path / "kernel_op.cu"
    source.write_text("int kernel = 1;\n", encoding="utf-8")
    private_output = tmp_path / "kernel_op.so.tmp.123"
    result = subprocess.run(
        [
            sys.executable,
            str(JITTOR / "utils" / "dlink_compiler.py"),
            str(fake_compiler),
            str(source),
            "-dc",
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
