"""An executor wait may drop the Python holders from which its roots came."""
import pytest

from _helpers.child_process import run_child_script


PROGRAM = r'''
import numpy as np
import jittor as jt
import jittor_utils

probe = jittor_utils.compile_module(r"""
#include "bindings/pyjt/py_converter.h"
#include "runtime/async_exec.h"
#include <condition_variable>
#include <mutex>
namespace jittor {
// @pyjt(queue_python_release)
bool queue_python_release(PyObject* callback) {
    async_exec_wait();
    if (!async_exec_placed()) return false;
    std::mutex mutex;
    std::condition_variable ready;
    bool queued = false;
    Py_INCREF(callback);
    async_exec_submit([&] {
        async_exec_defer_python_release([callback] {
            PyObjHolder result(PyObject_CallObject(callback, nullptr));
            Py_DECREF(callback);
            if (!result.obj) throw std::runtime_error("deferred root release failed");
        });
        std::lock_guard<std::mutex> lock(mutex);
        queued = true;
        ready.notify_one();
    });
    std::unique_lock<std::mutex> lock(mutex);
    // The callback must already be queued when sync_all collects its roots;
    // waiting for the executor here would drain it and hide the stale roots.
    ready.wait(lock, [&] { return queued; });
    return true;
}
}
""", jt.compiler.cc_flags + " " + jt.compiler.opt_flags)

with jt.flag_scope(auto_flush_ops=0, async_flush_ops=0, backend_fallback="error"):
    before = jt.core.async_exec_batches()
    for step in range(8):
        # These pending sinks are owned only by the list the callback clears,
        # just as a Function context owns its saved tensors. sync_all collects
        # them before its executor entry drains that callback.
        victims = [(jt.ones((7, 13)) + float(step)).sin() for _ in range(8)]
        values = np.arange(32, dtype="float32") + step
        source = jt.array(values).start_grad()
        survivor = source * 2.0
        if not probe.queue_python_release(victims.clear):
            print("ASYNC_ROOTS_NO_WORKER")
            raise SystemExit(0)
        jt.sync_all(True)
        assert not victims
        expected_location = "device" if jt.flags.use_cuda else "cpu"
        assert survivor.location() == expected_location
        np.testing.assert_array_equal(survivor.numpy(), values * 2.0)
        gradient = jt.grad(survivor.sum(), source)
        gradient.sync()
        assert gradient.location() == expected_location
        np.testing.assert_array_equal(gradient.numpy(), np.full_like(values, 2.0))
    assert jt.core.async_exec_batches() - before == 8
    assert jt.core.backend_fallback_count() == 0
print("ASYNC_ROOTS_OK")
'''


def test_sync_all_owns_roots_before_running_deferred_python_releases():
    result = run_child_script(PROGRAM, crash_isolated=True,
                              without_torch_mode=True)
    assert result.returncode == 0, result.stdout + result.stderr
    if b"ASYNC_ROOTS_NO_WORKER" in result.stdout:
        pytest.skip("requires two permitted CPUs in one cache domain")
    assert b"ASYNC_ROOTS_OK" in result.stdout
