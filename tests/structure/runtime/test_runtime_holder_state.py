"""Native holder ownership and weak-sync cursor lifecycle, without a JIT build."""

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"


def test_native_holder_state_cursor_lifecycle(tmp_path):
    source = tmp_path / "holder_state.cc"
    source.write_text(r'''
#include "runtime/runtime.h"
#include <cassert>
#include <type_traits>
namespace jittor {
struct VarHolder { int id; };
// Only LaunchHistory::report needs it, and nothing below calls report.
// Stubbed rather than linking backend.cc, which drags the whole backend
// registry into a case about holder-state ownership.
const char* backend_name(BackendId) { return "cpu"; }
}
using namespace jittor;
int main() {
    NativeRuntime isolated;
    assert(isolated.executor().allocator == nullptr);
    assert(isolated.executor().temp_allocator == nullptr);
    assert(!isolated.executor().last_is_cuda);
    // Submission scheduling is not executor state: it has its own owner.
    assert(!isolated.submissions().flush_active);
    assert(isolated.submissions().last_run_ops == 0);
    assert(&runtime_submission_pipeline() == &native_runtime().submissions());
    assert(&isolated.submissions() != &runtime_submission_pipeline());
    assert(&runtime_executor() == &native_runtime().executor());
    assert(&runtime_holder_state() == &native_runtime().holders());
    assert(&isolated.executor() != &runtime_executor());
    assert(&isolated.holders() != &runtime_holder_state());
    assert(&runtime_traversal_state() == &native_runtime().traversals());
    assert(&isolated.traversals() != &runtime_traversal_state());
    assert(isolated.traversals().stamp_count() == 0);
    assert(isolated.traversals().active_epochs() == 0);
    assert(&runtime_device_state() == &native_runtime().devices());
    assert(&isolated.devices() != &runtime_device_state());
    assert(isolated.devices().use_cuda == 0);
    assert(isolated.devices().device_id == -1);
    assert(isolated.devices().sync_run == 1);
    assert(isolated.devices().device_count == -1);
    assert(isolated.devices().current_device == -1);
    assert(isolated.devices().switch_hooks.empty());
    assert(isolated.devices().peer_enabled.empty());
    assert(&runtime_flag_use_cuda() == &runtime_device_state().use_cuda);
    assert(&runtime_flag_device_id() == &runtime_device_state().device_id);
    assert(&runtime_flag_sync_run() == &runtime_device_state().sync_run);
    static_assert(!std::is_copy_constructible<RuntimeHolderState>::value, "owner");
    static_assert(!std::is_move_constructible<RuntimeHolderState>::value, "cursor");
    RuntimeHolderState roots;
    assert(roots.holders().empty() && !roots.peek_pending());
    roots.consume_pending();
    VarHolder a{1}, b{2}, c{3}, moved{4};
    auto ia = roots.add(&a);
    auto ib = roots.add(&b);
    auto ic = roots.add(&c);
    assert(roots.peek_pending() == &a);
    roots.consume_pending();
    // Peeking beyond a sync cutoff must not consume the next candidate.
    assert(roots.peek_pending() == &b);
    assert(roots.peek_pending() == &b);
    roots.erase(ia); // erase the consumed cursor node, not the pending one
    roots.erase(ia); // released holder can subsequently be destructed
    assert(!roots.contains(ia) && roots.peek_pending() == &b);
    *ib = &moved; // VarHolder's move-from-pointer constructor keeps its slot
    assert(roots.peek_pending() == &moved);
    roots.erase(ib);
    assert(roots.peek_pending() == &c);
    roots.consume_pending();
    assert(!roots.peek_pending());
    auto another = roots.add(&a);
    assert(roots.peek_pending() == &a);
    roots.erase(ic);
    assert(roots.peek_pending() == &a);
    roots.erase(another);
    assert(roots.holders().empty() && !roots.peek_pending());
    assert(&runtime_holder_state() == &runtime_holder_state());
    return 0;
}
''', encoding="utf-8")
    executable = tmp_path / "holder_state"
    result = subprocess.run(
        [os.environ.get("CXX", "g++"), "-std=c++14", "-D_GLIBCXX_DEBUG",
         "-I", str(SRC), str(source), str(SRC / "runtime/holder_state.cc"),
         str(SRC / "runtime/runtime.cc"),
         # NativeRuntime owns a LaunchHistory, whose constructor and destructor
         # live here. Without it the link fails on a symbol that has nothing to
         # do with holder-state ownership, which is what this case is about.
         str(SRC / "runtime/launch_diagnostics.cc"),
         "-o", str(executable)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True, timeout=10)


def test_holder_globals_are_no_longer_exported():
    header = (SRC / "core/var_holder.h").read_text(encoding="utf-8")
    source = (SRC / "core/var_holder.cc").read_text(encoding="utf-8")
    assert "EXTERN_LIB list<VarHolder*> hold_vars" not in header
    assert "EXTERN_LIB list<VarHolder*>::iterator sync_ptr" not in header
    assert "list<VarHolder*> hold_vars;" not in source
    assert "runtime_holder_state().add(self)" in source


def test_executor_instance_is_owned_by_native_runtime():
    executor_header = (SRC / "core/executor.h").read_text(encoding="utf-8")
    executor_source = (SRC / "core/executor.cc").read_text(encoding="utf-8")
    assert "EXTERN_LIB Executor exe;" not in executor_header
    assert "Executor exe;" not in executor_source
    assert "EXTERN_LIB Executor& runtime_executor();" in executor_header


def test_executor_header_carries_no_submission_pipeline_state():
    """Scheduling bookkeeping belongs to the pipeline, not to the executor.

    `last_run_ops` and `flush_active` are inputs to *when* a graph is
    submitted, never to *how* it runs.  While they sat on `Executor` a reader
    of `executor.h` could not tell them apart from the allocators `run_sync`
    actually uses, and each new scheduling heuristic added another such field.
    """
    executor_header = (SRC / "core/executor.h").read_text(encoding="utf-8")
    pipeline_header = (SRC / "runtime/submission_pipeline.h").read_text(encoding="utf-8")
    for field in ("last_run_ops", "flush_active"):
        assert field not in executor_header, field
        assert field in pipeline_header, field
    assert "SubmissionPipeline submissions_;" in \
        (SRC / "runtime/runtime.h").read_text(encoding="utf-8")


def test_traversal_storage_and_implementation_belong_to_runtime():
    assert not (SRC / "misc/traversal_epoch.h").exists()
    assert not (SRC / "misc/traversal_epoch.cc").exists()
    node = (SRC / "core/node.h").read_text(encoding="utf-8")
    epoch = (SRC / "runtime/traversal_epoch.cc").read_text(encoding="utf-8")
    assert "EXTERN_LIB int64 tflag_count" not in node
    assert "TraversalEpoch::live_count" not in epoch
    assert "runtime_traversal_state()" in epoch


def test_device_header_is_sdk_free_and_callbacks_belong_to_native_providers():
    for defines in ([], ["-DIS_ROCM"]):
        result = subprocess.run(
            [os.environ.get("CXX", "g++"), "-std=c++14", "-DHAS_ACCELERATOR",
             *defines, "-I", str(SRC), "-fsyntax-only", "-x", "c++", "-"],
            input='#include "runtime/device.h"\n',
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
    device_header = (SRC / "runtime/device.h").read_text(encoding="utf-8")
    assert "_cudaLaunchHostFunc" not in device_header
    for backend, expected in (("cuda", "cudaLaunchHostFunc"), ("rocm", "hipLaunchHostFunc")):
        driver = (ROOT / "backends" / backend / "runtime/driver.cc").read_text(encoding="utf-8")
        callback = driver[driver.index("void host_callback("):]
        callback = callback[:callback.index("\n}") + 2] if "\n}" in callback else callback
        assert expected in callback
        assert "ops.host_callback = host_callback" in driver
