"""Run the real native ring without importing or compiling Jittor's core."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_launch_history_is_bounded_owned_and_allocation_free(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "launch_history.cc"
    source.write_text(r'''
#include "runtime/launch_diagnostics.h"
#include "runtime/device_state.h"
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
std::atomic<bool> forbid_allocation{false};
void* operator new(std::size_t n) {
    assert(!forbid_allocation.load());
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
namespace jittor {
LaunchHistory& runtime_launch_history() { static auto* h = new LaunchHistory; return *h; }
RuntimeDeviceState& runtime_device_state() { throw std::runtime_error("unused device stub"); }
const char* backend_name(BackendId id) { return id == BackendId::Cuda ? "cuda" : "cpu"; }
const BackendOps& backend_ops(BackendId) { throw std::runtime_error("unused backend stub"); }
BackendRegistry& backend_registry() { throw std::runtime_error("unused registry stub"); }
const BackendOps& BackendRegistry::get(const string&) const { throw std::runtime_error("unused lookup stub"); }
}
int main() {
    using namespace jittor;
    auto& history = runtime_launch_history();
    LaunchRecord r;
    char filename[] = "creation.py";
    r.origin = history.intern_origin(filename, 123);
    filename[0] = 'X'; // registry cannot borrow caller storage
    r.device = {BackendId::Cuda, 0};
    r.stream = 11;
    r.op_id = 91;
    std::strcpy(r.name, "original_op");
    history.record(r); // initialize this thread's ring
    forbid_allocation = true;
    for (int i=0; i<1000; ++i) history.record(r);
    forbid_allocation = false;
    auto text = history.report(r.device, true, 11);
    assert(text.find("creation.py:123") != string::npos);
    assert(text.find("seq=1001 ") < text.find("seq=1000 "));
    assert(text.find("overwritten=937") != string::npos);
    assert(text.find("seq=1 ") == string::npos);
    assert(history.report({BackendId::Cuda, 1}).find("not-found") != string::npos);
    assert(history.report(r.device, true, 12).find("not-found") != string::npos);
    std::atomic<int> ready{0};
    auto worker = [&](uintptr_t stream) {
        auto value = r;
        value.stream = stream;
        history.record(value);
        ++ready;
        while (ready.load() != 2) std::this_thread::yield();
        for (int i=0; i<100; ++i) history.record(value);
    };
    std::thread a(worker, 21), b(worker, 22);
    a.join(); b.join();
    assert(history.report(r.device, true, 21).find("creation.py:123") != string::npos);
    assert(history.report(r.device, true, 22).find("creation.py:123") != string::npos);
    assert(history.report(r.device).find("not proof") != string::npos);
    set_launch_origin_capture(+[]() -> uint64 { return 77; });
    assert(capture_launch_origin() == 77);
    {
        LaunchOriginScope unknown(0);
        assert(capture_launch_origin() == 0);
        { LaunchOriginScope known(9); assert(capture_launch_origin() == 9); }
        assert(capture_launch_origin() == 0);
    }
    assert(capture_launch_origin() == 77);
    set_launch_origin_capture(nullptr);
    for (size_t i=1; i<LaunchHistory::origin_capacity; ++i)
        assert(history.intern_origin("dynamic.py", int(i)) != 0);
    assert(history.intern_origin("overflow.py", 1) == 0);
    assert(history.intern_origin("creation.py", 123) == r.origin);
    assert(history.report(r.device).find("unrecorded_origin_captures=1") != string::npos);
    assert(history.intern_origin(string(8192, 'x').c_str(), 1) == 0);
    assert(history.report(r.device).find("unrecorded_origin_captures=2") != string::npos);
    assert(async_launch_history("acl", 0, -1).find("not-found") != string::npos);
    for (int which=0; which<3; ++which) {
        bool rejected = false;
        try { async_launch_history(which == 0 ? "invalid" : "cuda", which == 1 ? -1 : 0,
                                   which == 2 ? -2 : -1); }
        catch (const UserError&) { rejected = true; }
        assert(rejected);
    }
    // A thread lease keeps only native implementation storage alive. It must
    // not call a destroyed public LaunchHistory object on thread exit.
    std::atomic<int> state{0};
    auto transient = std::unique_ptr<LaunchHistory>(new LaunchHistory);
    auto* borrowed = transient.get();
    std::thread exiting([&] {
        borrowed->record(r);
        state = 1;
        while (state.load() != 2) std::this_thread::yield();
    });
    while (state.load() != 1) std::this_thread::yield();
    transient.reset();
    state = 2;
    exiting.join();
}
''', encoding="utf-8")
    binary = tmp_path / "launch_history"
    subprocess.run([compiler, "-std=c++14", "-pthread", "-I", str(root / "src"),
                    str(source), str(root / "src/runtime/launch_diagnostics.cc"),
                    "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, timeout=10)
