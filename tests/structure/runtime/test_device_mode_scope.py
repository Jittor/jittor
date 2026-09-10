"""Exercise native rollback snapshots without loading an accelerator or JIT."""
from pathlib import Path
import shutil
import subprocess
import sysconfig

import pytest


def test_native_device_mode_snapshot_restores_only_saved_state(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "device_mode_scope.cc"
    source.write_text(r'''
#include "runtime/device.h"
#include "mem/allocator.h"
#include <cassert>
#include <stdexcept>
namespace jittor {
RuntimeDeviceState::RuntimeDeviceState() = default;
RuntimeDeviceState::~RuntimeDeviceState() = default;
RuntimeDeviceState& runtime_device_state() {
    static RuntimeDeviceState state;
    return state;
}
}
int main() {
    using namespace jittor;
    auto& state = runtime_device_state();
    state.device_id = 3;
    state.current_device = 3;
    const auto outer = push_device_mode_scope();
    state.use_cuda = 1; // stand in for a successful, validated public setter
    const auto inner = push_device_mode_scope();
    state.use_cuda = 0;
    try {
        pop_device_mode_scope(inner + 1, true); // never-issued token
        assert(false);
    } catch (const std::invalid_argument&) {}
    assert(state.use_cuda == 0 && state.mode_scope_snapshots.size() == 2);
    pop_device_mode_scope(inner, true);
    assert(state.use_cuda == 1);
    pop_device_mode_scope(outer, true);
    assert(state.use_cuda == 0 && state.mode_scope_snapshots.empty());
    assert(state.device_id == 3 && state.current_device == 3);
    try {
        pop_device_mode_scope(outer, true);
        assert(false);
    } catch (const std::invalid_argument&) {}
    // Normal release cannot silently perform an exceptional rollback.
    const auto released = push_device_mode_scope();
    state.use_cuda = 2;
    pop_device_mode_scope(released, false);
    assert(state.use_cuda == 2 && state.mode_scope_snapshots.empty());
    // Deterministic interleaving of two Python threads using the same mode:
    // A enters, B enters, A exits, then B exits. GIL-held native calls are
    // individually serialized, but the scopes are not globally nested.
    const auto thread_a = push_device_mode_scope();
    const auto thread_b = push_device_mode_scope();
    pop_device_mode_scope(thread_a, false);
    assert(state.use_cuda == 2 && state.mode_scope_snapshots.size() == 1);
    assert(state.mode_scope_snapshots[0].token == thread_b);
    pop_device_mode_scope(thread_b, false);
    assert(state.use_cuda == 2 && state.mode_scope_snapshots.empty());
    // Exceptional exits also recover the value owned by their token even
    // when another active scope was entered more recently.
    const auto first = push_device_mode_scope();
    state.use_cuda = 1;
    const auto second = push_device_mode_scope();
    state.use_cuda = 0;
    pop_device_mode_scope(first, true);
    assert(state.use_cuda == 2 && state.mode_scope_snapshots.size() == 1);
    pop_device_mode_scope(second, true);
    assert(state.use_cuda == 1 && state.mode_scope_snapshots.empty());
}
''')
    executable = tmp_path / "device_mode_scope"
    subprocess.run([
        compiler, "-std=c++14", "-I" + str(root / "src"),
        "-I" + sysconfig.get_path("include"), str(source),
        str(root / "src/runtime/device_mode_scope.cc"), "-o", str(executable),
    ], check=True, capture_output=True, text=True, timeout=30)
    subprocess.run([str(executable)], check=True, capture_output=True,
                   text=True, timeout=10)
