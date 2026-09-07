"""Shared executor/diagnostics compile without any accelerator SDK headers."""

import os
from pathlib import Path
import re
import subprocess
import sysconfig


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CONSUMERS = (
    "core/executor.cc", "core/exec_plan.cc", "core/exec_runner.cc",
    "runtime/init.cc", "core/event_queue.cc", "runtime/profiler/profiler.cc",
    "debug/nan_checker.cc", "utils/log.cc", "bindings/pyjt/py_converter.h",
)


def test_shared_device_consumers_have_no_vendor_sdk_dependency():
    for relative in CONSUMERS + ("core/executor.h", "core/exec_plan.h", "core/exec_runner.h",
                                 "core/event_queue.h", "debug/nan_checker.h"):
        source = (SRC / relative).read_text(encoding="utf-8")
        assert not re.search(r"#\s*include\s*[<\"](?:cuda|hip|acl/|helper_cuda)", source), relative
        assert not re.search(r"\b(?:cuda|hip)[A-Z]\w*\s*\(", source), relative
        if relative == "bindings/pyjt/py_converter.h":
            assert "#ifdef IS_CUDA" not in source
            assert "accelerator_backend_id() == BackendId::Cuda" in source
    command = [os.environ.get("CXX", "g++"), "-std=c++14", "-fsyntax-only",
               "-DHAS_ACCELERATOR", "-I" + str(SRC),
               "-I" + sysconfig.get_path("include")]
    result = subprocess.run(command + [str(SRC / name) for name in CONSUMERS],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_auto_flush_requires_backend_support_and_nan_diagnostics_dispatch():
    executor = (SRC / "core/executor.cc").read_text(encoding="utf-8")
    diagnostics = (SRC / "debug/nan_checker.cc").read_text(encoding="utf-8")
    assert "execution.supports_auto_flush" in executor
    assert "CHECK(backend.check_nan)" in diagnostics
    assert "backend.check_nan(v, op)" in diagnostics
    assert "allocation_device(v->allocator)" in diagnostics
    assert (ROOT / "backends/cuda/runtime/nan_checker.cc").is_file()
