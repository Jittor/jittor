"""Backend callbacks observed through real tensor execution, not registry metadata."""

import numpy as np
import pytest
import warnings


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


def test_cpu_tensor_execution_uses_registered_allocator():
    import jittor as jt

    assert "cpu" in jt.core.registered_backends()
    assert jt.core.backend_device_count("cpu") == 1
    probe = _backend_probe(jt)
    data = np.arange(33, dtype=np.float32)
    with jt.flag_scope(use_cuda=0, use_cuda_host_allocator=0):
        jt.sync_all(True)
        probe.native_backend_probe_install(0)
        try:
            value = jt.array(data) * 3 + 7
            np.testing.assert_array_equal(value.numpy(), data * 3 + 7)
            assert value.location() == "cpu"
            jt.sync_all(True)
            assert probe.native_backend_probe_counts()[0] > 0
            assert probe.native_backend_probe_counts()[1] > 0
            assert probe.native_backend_probe_counts()[3] > 0
        finally:
            probe.native_backend_probe_restore()
        # Restoring the table must leave ordinary execution usable.
        np.testing.assert_array_equal((value + 1).numpy(), data * 3 + 8)


def test_unknown_backend_name_is_rejected():
    import jittor as jt

    with pytest.raises(RuntimeError, match="[Bb]ackend|registered"):
        jt.core.backend_device_count("missing_backend_for_registry_test")


@pytest.mark.parametrize("name", ["use_device", "use_acl", "use_rocm", "use_corex"])
def test_legacy_backend_aliases_warn_and_preserve_setter_semantics(name):
    import jittor as jt

    with jt.runtime.scope(use_cuda=0):
        with pytest.warns(DeprecationWarning, match="deprecated accelerator-mode alias"):
            assert getattr(jt.flags, name) == 0
        with pytest.warns(DeprecationWarning, match="deprecated accelerator-mode alias"):
            setattr(jt.flags, name, False)
        assert jt.runtime.use_cuda == 0


def test_warning_as_error_does_not_change_backend_mode():
    import jittor as jt

    with jt.runtime.scope(use_cuda=0), warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match="deprecated accelerator-mode alias"):
            jt.flags.use_device = True
        assert jt.runtime.use_cuda == 0
