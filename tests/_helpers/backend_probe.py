"""Shared native BackendOps instrumentation for CPU and CUDA tests."""

_PROBE_HEADER = r"""
#pragma once
#include "runtime/backend.h"
namespace jittor {
// @pyjt(native_backend_probe_install)
void native_backend_probe_install(int accelerator);
// @pyjt(native_backend_probe_restore)
void native_backend_probe_restore();
// @pyjt(native_backend_probe_counts)
vector<int> native_backend_probe_counts();
}
"""

_PROBE_SOURCE = r"""
#include "native_backend_probe.h"
#include <stdexcept>
namespace jittor {
namespace {
BackendOps saved;
BackendOps* observed = nullptr;
int allocations = 0, copies = 0, async_copies = 0, syncs = 0;
int uploads = 0, downloads = 0;
Allocator* count_allocator(int device, BackendMemoryKind kind) {
    ++allocations;
    return saved.allocator(device, kind);
}
void count_direction(Device dst, Device src) {
    if (dst.backend != BackendId::Cpu && src.backend == BackendId::Cpu) ++uploads;
    if (dst.backend == BackendId::Cpu && src.backend != BackendId::Cpu) ++downloads;
}
void count_copy(void* dst, Device dd, const void* src, Device sd,
                size_t size, bool ordered) {
    ++copies;
    count_direction(dd, sd);
    saved.copy(dst, dd, src, sd, size, ordered);
}
void count_copy_async(void* dst, Device dd, const void* src, Device sd,
                      size_t size, BackendStream stream) {
    ++async_copies;
    count_direction(dd, sd);
    saved.copy_async(dst, dd, src, sd, size, stream);
}
void count_sync(uint64 mask) {
    ++syncs;
    saved.synchronize(mask);
}
}
void native_backend_probe_install(int accelerator) {
    if (observed) throw std::runtime_error("backend probe already installed");
    auto id = accelerator ? accelerator_backend_id() : BackendId::Cpu;
    observed = &const_cast<BackendOps&>(backend_ops(id));
    saved = *observed;
    allocations = copies = async_copies = syncs = uploads = downloads = 0;
    observed->allocator = count_allocator;
    observed->copy = count_copy;
    observed->copy_async = count_copy_async;
    observed->synchronize = count_sync;
}
void native_backend_probe_restore() {
    if (!observed) return;
    *observed = saved;
    observed = nullptr;
}
vector<int> native_backend_probe_counts() {
    return {allocations, copies, async_copies, syncs, uploads, downloads};
}
}
"""


def _backend_probe(jt):
    import jittor_utils

    source = _PROBE_HEADER + _PROBE_SOURCE.replace('#include "native_backend_probe.h"', "")
    return jittor_utils.compile_module(source, jt.compiler.cc_flags)
