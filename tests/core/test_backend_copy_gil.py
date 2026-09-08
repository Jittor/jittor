"""Blocking copy releases the GIL only while the executor owns its entry lock."""
import threading
import time

import pytest


@pytest.fixture(scope="module")
def copy_probe():
    import jittor as jt
    import jittor_utils
    return jittor_utils.compile_module(r'''
#include "runtime/backend.h"
#include "runtime/executor_entry.h"
#include "bindings/pyjt/gil.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <stdexcept>
namespace jittor {
namespace {
std::atomic<bool> waiting{false}, acknowledged{false};
int held_during = -1;
bool inject_failure = false;
void controlled_copy(void*, Device, const void*, Device, size_t, bool) {
    held_during = PyGILState_Check();
    waiting.store(true);
    if (!held_during) {
        const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!acknowledged.load() && std::chrono::steady_clock::now() < limit)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (inject_failure) throw std::runtime_error("injected copy failure");
}
}
// @pyjt(reset_probe)
void reset_probe() { waiting.store(false); acknowledged.store(false); }
// @pyjt(copy_waiting)
bool copy_waiting() { return waiting.load(); }
// @pyjt(acknowledge)
void acknowledge() { acknowledged.store(true); }
// @pyjt(run_copy)
vector<int> run_copy(bool own_executor, bool fail) {
    auto& ops = const_cast<BackendOps&>(backend_ops(accelerator_backend_id()));
    const auto original = ops.copy;
    ops.copy = controlled_copy;
    inject_failure = fail;
    int caught = 0;
    char src = 1, dst = 0;
    try {
        if (own_executor) {
            ExecutorEntryScope outer;
            ExecutorEntryScope nested;
            backend_copy(&dst, {BackendId::Cpu, 0}, &src,
                         {accelerator_backend_id(), 0}, 1);
        } else {
            backend_copy(&dst, {BackendId::Cpu, 0}, &src,
                         {accelerator_backend_id(), 0}, 1);
        }
    } catch (const std::runtime_error&) { caught = 1; }
    ops.copy = original;
    return {held_during, acknowledged.load() ? 1 : 0,
            PyGILState_Check(), caught, inside_executor() ? 1 : 0};
}
}''', jt.compiler.cc_flags)


@pytest.mark.parametrize("fail", [False, True])
def test_copy_wait_allows_python_handshake_and_restores_gil(copy_probe, fail):
    copy_probe.reset_probe()
    stop = threading.Event()

    def acknowledge_from_python():
        while not stop.is_set():
            if copy_probe.copy_waiting():
                copy_probe.acknowledge()
                return
            time.sleep(0.001)

    worker = threading.Thread(target=acknowledge_from_python)
    worker.start()
    try:
        result = copy_probe.run_copy(True, fail)
    finally:
        stop.set()
        worker.join(timeout=3)
    assert not worker.is_alive()
    assert result == [0, 1, 1, int(fail), 0]


def test_copy_outside_executor_keeps_gil(copy_probe):
    copy_probe.reset_probe()
    assert copy_probe.run_copy(False, False) == [1, 0, 1, 0, 0]
