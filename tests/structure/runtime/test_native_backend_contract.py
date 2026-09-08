"""Compile the registry in isolation: no device SDK, Python core, or JIT cache."""

import os
from pathlib import Path
import subprocess

import pytest


SRC = Path(__file__).resolve().parents[3] / "src"


@pytest.mark.parametrize("legacy_define", [[], ["-DCUDA=ACL"]])
def test_backend_registry_validates_and_owns_its_callback_table(tmp_path, legacy_define):
    source = tmp_path / "backend_registry_contract.cc"
    source.write_text(r'''
#include "runtime/backend.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <type_traits>
using namespace jittor;
namespace {
int calls = 0;
int count() { ++calls; return 3; }
int current() { ++calls; return 0; }
void set(int) { ++calls; }
Allocator* allocate(int, BackendMemoryKind) { ++calls; return nullptr; }
void copy(void*, Device, const void*, Device, size_t, bool) { ++calls; }
void copy_async(void*, Device, const void*, Device, size_t, BackendStream) { ++calls; }
void sync(uint64) { ++calls; }
void* stream(int, BackendStreamKind) { ++calls; return nullptr; }
void peer(int, int) { ++calls; }
BackendOps complete() {
    BackendOps ops;
    ops.name = "cpu";
    ops.device_count = count;
    ops.current_device = current;
    ops.set_device = set;
    ops.allocator = allocate;
    ops.copy = copy;
    ops.copy_async = copy_async;
    ops.synchronize = sync;
    ops.stream = stream;
    ops.enable_peer = peer;
    ops.memory_allocate = [](int, BackendMemoryKind, size_t) -> void* { ++calls; return nullptr; };
    ops.memory_free = [](int, BackendMemoryKind, void*) { ++calls; };
    ops.memory_info = [](int, size_t&, size_t&) { ++calls; };
    ops.check_error = [] { ++calls; };
    ops.compute_stream = [](int) -> void* { ++calls; return nullptr; };
    ops.stream_create = [](int, bool) -> void* { ++calls; return nullptr; };
    ops.stream_destroy = [](BackendStream) { ++calls; };
    ops.stream_synchronize = [](BackendStream) { ++calls; };
    ops.event_create = [](int, bool) -> void* { ++calls; return nullptr; };
    ops.event_destroy = [](BackendEvent) { ++calls; };
    ops.event_record = [](BackendEvent, BackendStream) { ++calls; };
    ops.event_synchronize = [](BackendEvent) { ++calls; };
    ops.event_elapsed = [](BackendEvent, BackendEvent) { ++calls; return 0.f; };
    ops.stream_wait_event = [](BackendStream, BackendEvent) { ++calls; };
    ops.host_callback = [](BackendStream, void (*)(void*), void*) { ++calls; };
    return ops;
}
template<class F> void invalid(F action) {
    bool rejected = false;
    try { action(); }
    catch (const std::invalid_argument& error) {
        rejected = true;
        assert(std::strlen(error.what()) > 0);
    }
    assert(rejected);
}
void reject(BackendOps ops) {
    BackendRegistry registry;
    invalid([&] { registry.register_backend(ops); });
    assert(registry.names().empty());
    registry.register_backend(complete());
    assert(registry.names().size() == 1);
}
}
int main() {
    static_assert(!std::is_copy_constructible<BackendRegistry>::value, "owner");
    auto bad = complete(); bad.abi_version = 0; reject(bad);
    bad = complete(); bad.abi_version = 1; reject(bad);
    bad = complete(); bad.struct_size = 0; reject(bad);
    bad = complete(); bad.struct_size = offsetof(BackendOps, execution); reject(bad);
    bad = complete(); bad.name = nullptr; reject(bad);
    bad = complete(); bad.name = ""; reject(bad);
    bad = complete(); bad.device_count = nullptr; reject(bad);
    bad = complete(); bad.current_device = nullptr; reject(bad);
    bad = complete(); bad.set_device = nullptr; reject(bad);
    bad = complete(); bad.allocator = nullptr; reject(bad);
    bad = complete(); bad.copy = nullptr; reject(bad);
    bad = complete(); bad.copy_async = nullptr; reject(bad);
    bad = complete(); bad.synchronize = nullptr; reject(bad);
    bad = complete(); bad.stream = nullptr; reject(bad);
    bad = complete(); bad.enable_peer = nullptr; reject(bad);
    bad = complete(); bad.memory_allocate = nullptr; reject(bad);
    bad = complete(); bad.memory_free = nullptr; reject(bad);
    bad = complete(); bad.memory_info = nullptr; reject(bad);
    bad = complete(); bad.check_error = nullptr; reject(bad);
    bad = complete(); bad.compute_stream = nullptr; reject(bad);
    bad = complete(); bad.stream_create = nullptr; reject(bad);
    bad = complete(); bad.stream_destroy = nullptr; reject(bad);
    bad = complete(); bad.stream_synchronize = nullptr; reject(bad);
    bad = complete(); bad.event_create = nullptr; reject(bad);
    bad = complete(); bad.event_destroy = nullptr; reject(bad);
    bad = complete(); bad.event_record = nullptr; reject(bad);
    bad = complete(); bad.event_synchronize = nullptr; reject(bad);
    bad = complete(); bad.event_elapsed = nullptr; reject(bad);
    bad = complete(); bad.stream_wait_event = nullptr; reject(bad);
    bad = complete(); bad.host_callback = nullptr; reject(bad);
    BackendRegistry registry;
    char mutable_name[] = "cpu";
    auto ops = complete();
    ops.name = mutable_name;
    assert(ops.execution.supports_parallel_compile);
    assert(!ops.execution.requires_pinned_host_storage);
    assert(!ops.execution.preserve_reduction_dtype);
    assert(!ops.execution.native_low_precision_reduction);
    registry.register_backend(ops);
    mutable_name[0] = 'X';
    ops.device_count = nullptr;
    ops.execution.supports_parallel_compile = false;
    assert(registry.get(BackendId::Cpu).execution.supports_parallel_compile);
    assert(registry.names() == vector<string>{"cpu"});
    assert(std::strcmp(registry.get(BackendId::Cpu).name, "cpu") == 0);
    assert(registry.get(BackendId::Cpu).device_count == count);
    // Publishing a table does not initialize hardware or call an allocator.
    assert(calls == 0);
    assert(registry.get(BackendId::Cpu).device_count() == 3);
    assert(calls == 1);
    auto duplicate_id = complete(); duplicate_id.name = "another_cpu";
    invalid([&] { registry.register_backend(duplicate_id); });
    auto duplicate_name = complete(); duplicate_name.id = BackendId::Cuda;
    invalid([&] { registry.register_backend(duplicate_name); });
    assert(registry.names() == vector<string>{"cpu"});
    bool missing = false;
    try { registry.get(BackendId::Cuda); }
    catch (const std::out_of_range&) { missing = true; }
    assert(missing);
    auto cuda = complete(); cuda.name = "cuda"; cuda.id = BackendId::Cuda;
    registry.register_backend(cuda);
    assert(registry.names().size() == 2);
    auto acl = complete(); acl.name = "acl"; acl.id = BackendId::Acl;
    registry.register_backend(acl);
    assert(&registry.get("acl") == &registry.get("acl_legacy"));
    assert(registry.names().size() == 3);
    assert(registry.get(BackendId::Cpu).device_count == count);
    assert(std::strcmp(registry.get(BackendId::Cpu).name, "cpu") == 0);
}
''', encoding="utf-8")
    executable = tmp_path / "backend_registry_contract"
    result = subprocess.run(
        [os.environ.get("CXX", "g++"), "-std=c++14", *legacy_define, "-I", str(SRC),
         str(source), str(SRC / "runtime/backend_registry.cc"),
         "-o", str(executable)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True, timeout=10)
