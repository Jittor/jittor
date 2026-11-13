"""Real stream gates keep DMA in flight; no timing-based speed assertions."""
import gc

import jittor as jt
import jittor_utils
import numpy as np
import pytest

# Both cases require a real accelerator; expose that to the gate counter.
pytestmark = pytest.mark.npu if getattr(jt.compiler, "has_acl", False) else pytest.mark.cuda


@pytest.fixture(scope="module")
def stream_gate():
    if not (jt.has_cuda or getattr(jt.compiler, "has_acl", False)):
        pytest.skip("real accelerator required")
    return jittor_utils.compile_module(r'''
#include "runtime/backend.h"
#include <atomic>
#include <chrono>
#include <thread>
namespace jittor {
namespace {
std::atomic<bool> entered{false}, released{false}, expired{false};
BackendStream gated_stream;
void block_stream(void*) {
    entered.store(true);
    auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!released.load()) {
        if (std::chrono::steady_clock::now() >= limit) { expired.store(true); break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
}
// @pyjt(start_gate)
void start_gate(bool copy) {
    entered.store(false); released.store(false); expired.store(false);
    auto& ops = backend_ops(accelerator_backend_id());
    gated_stream = backend_stream({ops.id, ops.current_device()},
        copy ? BackendStreamKind::Copy : BackendStreamKind::Compute);
    ops.host_callback(gated_stream, block_stream, nullptr);
    auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!entered.load()) {
        if (std::chrono::steady_clock::now() >= limit) {
            released.store(true);
            USER_CHECK(false) << "native stream gate did not start";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
// @pyjt(release_gate)
void release_gate() { released.store(true); }
// @pyjt(gate_expired)
bool gate_expired() { return expired.load(); }
// @pyjt(drain_gate)
void drain_gate() { backend_ops(gated_stream.device.backend).stream_synchronize(gated_stream); }
}''', jt.compiler.cc_flags)


def host(values):
    with jt.flag_scope(use_cuda=0):
        value = jt.array(values).stop_grad()
        value.sync()
        return value


def device_empty(shape):
    value = jt.empty(shape, dtype="float32").stop_grad()
    value.sync(False, False)
    assert value.location() == "device"
    return value


def test_inflight_copy_survives_holder_destruction_and_rebind(stream_gate):
    with jt.flag_scope(use_cuda=1, backend_fallback="error", auto_graph_replay=0):
        values = np.arange(32768, dtype=np.float32)
        # Warm spare host blocks before the gate: growing a pinned-memory pool
        # may itself synchronize a driver, which is not the copy API's contract.
        warm = [host(values) for _ in range(4)]
        del warm
        src = host(values)
        dst, replacement = device_empty(values.shape), device_empty(values.shape)
        original = dst.detach(); original.sync(False, False)
        jt.sync_all(True)
        stream_gate.start_gate(True)
        try:
            jt.core._copy_into_async(dst, src)
            del src
            gc.collect()
            pressure = host(np.full(values.shape, -9, dtype=np.float32))
            dst.assign(replacement)
            assert not stream_gate.gate_expired(), "enqueue/rebind drained the copy stream"
        finally:
            stream_gate.release_gate()
        jt.core._copy_wait(original)
        stream_gate.drain_gate()
        np.testing.assert_array_equal(original.numpy(), values)
        np.testing.assert_array_equal(pressure.numpy(), np.full(values.shape, -9, dtype=np.float32))


def test_copy_wait_does_not_drain_unrelated_compute(stream_gate):
    with jt.flag_scope(use_cuda=1, backend_fallback="error", auto_graph_replay=0):
        values = np.arange(32768, dtype=np.float32)
        src, dst = host(values), device_empty(values.shape)
        # Initialize the copy stream and event machinery before blocking compute.
        jt.core._copy_into_async(dst, src); jt.core._copy_wait(dst)
        jt.sync_all(True)
        stream_gate.start_gate(False)
        try:
            jt.core._copy_into_async(dst, src)
            jt.core._copy_wait(dst)
            assert not stream_gate.gate_expired(), "copy_wait synchronized unrelated compute"
        finally:
            stream_gate.release_gate()
            stream_gate.drain_gate()
        np.testing.assert_array_equal(dst.numpy(), values)
