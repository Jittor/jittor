"""Native holder ownership and weak-sync cursor lifecycle, without a JIT build."""

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "python/jittor/src"


def test_native_holder_state_cursor_lifecycle(tmp_path):
    source = tmp_path / "holder_state.cc"
    source.write_text(r'''
#include "runtime/runtime.h"
#include <cassert>
#include <type_traits>
namespace jittor { struct VarHolder { int id; }; }
using namespace jittor;
int main() {
    NativeRuntime isolated;
    assert(isolated.executor().allocator == nullptr);
    assert(isolated.executor().temp_allocator == nullptr);
    assert(!isolated.executor().flush_active);
    assert(!isolated.executor().last_is_cuda);
    assert(isolated.executor().last_run_ops == 0);
    assert(&runtime_executor() == &native_runtime().executor());
    assert(&runtime_holder_state() == &native_runtime().holders());
    assert(&isolated.executor() != &runtime_executor());
    assert(&isolated.holders() != &runtime_holder_state());
    assert(&runtime_traversal_state() == &native_runtime().traversals());
    assert(&isolated.traversals() != &runtime_traversal_state());
    assert(isolated.traversals().stamp_count() == 0);
    assert(isolated.traversals().active_epochs() == 0);
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
         "-o", str(executable)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True, timeout=10)


def test_holder_globals_are_no_longer_exported():
    header = (SRC / "var_holder.h").read_text(encoding="utf-8")
    source = (SRC / "var_holder.cc").read_text(encoding="utf-8")
    assert "EXTERN_LIB list<VarHolder*> hold_vars" not in header
    assert "EXTERN_LIB list<VarHolder*>::iterator sync_ptr" not in header
    assert "list<VarHolder*> hold_vars;" not in source
    assert "runtime_holder_state().add(self)" in source


def test_executor_instance_is_owned_by_native_runtime():
    executor_header = (SRC / "executor.h").read_text(encoding="utf-8")
    executor_source = (SRC / "executor.cc").read_text(encoding="utf-8")
    assert "EXTERN_LIB Executor exe;" not in executor_header
    assert "Executor exe;" not in executor_source
    assert "EXTERN_LIB Executor& runtime_executor();" in executor_header


def test_traversal_storage_and_implementation_belong_to_runtime():
    assert not (SRC / "misc/traversal_epoch.h").exists()
    assert not (SRC / "misc/traversal_epoch.cc").exists()
    node = (SRC / "node.h").read_text(encoding="utf-8")
    epoch = (SRC / "runtime/traversal_epoch.cc").read_text(encoding="utf-8")
    assert "EXTERN_LIB int64 tflag_count" not in node
    assert "TraversalEpoch::live_count" not in epoch
    assert "runtime_traversal_state()" in epoch
