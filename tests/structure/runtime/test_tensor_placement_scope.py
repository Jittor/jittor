"""Construction placement is scoped state, independent of driver initialization."""
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tempfile

import pytest


def test_native_tensor_placement_scope_nesting_exception_and_thread_isolation():
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("backend prerequisite: C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    source = r'''
#include "runtime/tensor_placement.h"
#include <cassert>
#include <stdexcept>
#include <thread>
using namespace jittor;
int main() {
    assert(!current_tensor_placement().explicit_backend);
    {
        TensorPlacementScope cpu(TensorPlacement({BackendId::Cpu, 0}));
        assert(current_tensor_placement().device.backend == BackendId::Cpu);
        std::thread independent([] {
            assert(!current_tensor_placement().explicit_backend);
            TensorPlacementScope acl(TensorPlacement({BackendId::Acl, 2}));
            assert(current_tensor_placement().device.index == 2);
        });
        independent.join();
        try {
            TensorPlacementScope cuda(TensorPlacement({BackendId::Cuda, 1}));
            assert(current_tensor_placement().device.backend == BackendId::Cuda);
            throw std::runtime_error("unwind placement");
        } catch (const std::runtime_error&) {}
        assert(current_tensor_placement().device.backend == BackendId::Cpu);
        assert(current_tensor_placement().device.index == 0);
    }
    assert(!current_tensor_placement().explicit_backend);
}
'''
    with tempfile.TemporaryDirectory(prefix="jittor-placement-") as temporary:
        directory = Path(temporary)
        case = directory / "placement.cc"
        executable = directory / "placement"
        case.write_text(source)
        subprocess.run([
            compiler, "-std=c++14", "-pthread", "-I" + str(root / "src"),
            "-I" + sysconfig.get_path("include"), str(case),
            str(root / "src/runtime/tensor_placement.cc"), "-o", str(executable),
        ], check=True, capture_output=True, text=True, timeout=30)
        subprocess.run([str(executable)], check=True, capture_output=True,
                       text=True, timeout=10)
