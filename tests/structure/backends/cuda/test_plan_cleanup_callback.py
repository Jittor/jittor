"""A throwing allocator cannot escape through cuTT's destructor callback."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_cutt_free_callback_reports_failure_without_crossing_noexcept(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("backend prerequisite: C++ compiler unavailable")
    root = Path(__file__).resolve().parents[4]
    source = (root / "backends/cuda/libraries/cutt/src/cutt_wrapper.cc").read_text()
    declarations = source[source.index("struct CuttAllocation"):source.index("void jt_alloc")]
    callback = source[source.index("void jt_free"):source.index("int cutt_max_cache_size")]
    code = r'''
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <cassert>
using uint64 = unsigned long long;
#define LOGe std::cerr
struct Allocator {
    bool fail = false;
    int frees = 0;
    void free(void*, size_t, size_t&) {
        if (fail) throw std::runtime_error("injected allocator failure");
        ++frees;
    }
};
struct CacheDeviceScope { bool active = true; CacheDeviceScope(int, bool) {} };
''' + declarations + callback + r'''
struct VendorDestructor {
    void* pointer;
    ~VendorDestructor() noexcept {
        size_t allocation = 7;
        jt_free(pointer, 16, allocation);
    }
};
int main() {
    Allocator owner;
    owner.fail = true;
    void* pointer = reinterpret_cast<void*>(16);
    cutt_allocators.emplace(pointer, CuttAllocation{&owner, 1});
    { VendorDestructor vendor{pointer}; }
    assert(cutt_callback_failures == 1 && cutt_allocators.size() == 1);
    owner.fail = false;
    { VendorDestructor vendor{pointer}; }
    assert(owner.frees == 1 && cutt_allocators.empty());
    { VendorDestructor unknown{reinterpret_cast<void*>(32)}; }
    assert(cutt_callback_failures == 2 && owner.frees == 1);
}
'''
    unit = tmp_path / "callback.cc"
    executable = tmp_path / "callback"
    unit.write_text(code)
    built = subprocess.run([compiler, "-std=c++14", str(unit), "-o", str(executable)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "injected allocator failure" in result.stderr
