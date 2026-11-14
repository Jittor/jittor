"""An asynchronous job owns its construction flags, including nested setters."""
import unittest

import jittor_utils
import jittor as jt


class TestAsyncConstructionOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = jittor_utils.compile_module(r'''
#include "bindings/pyjt/py_converter.h"
#include "runtime/async_exec.h"
#include "core/node.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
namespace jittor {
DECLARE_RUNTIME_FLAG(uint8, node_order);
// @pyjt(check_construction_order)
vector<int64> check_construction_order(bool fail) {
    async_exec_wait();
    if (!async_exec_placed()) return {};
    auto previous = runtime_flag_node_order();
    set_node_order(0);
    std::mutex mutex;
    std::condition_variable ready;
    bool started = false, changed = false, timed_out = false;
    vector<int64> seen(5, -1);
    async_exec_submit([&] {
        seen[0] = runtime_flag_node_order();
        {
            std::unique_lock<std::mutex> lock(mutex);
            started = true;
            ready.notify_one();
            // A deadlock safety net, not a performance budget. A setter that
            // waits for this job cannot complete the handoff it requires.
            timed_out = !ready.wait_for(lock, std::chrono::seconds(60), [&] { return changed; });
        }
        seen[1] = runtime_flag_node_order();
        set_node_order(2);
        seen[2] = runtime_flag_node_order();
        Node node;
        seen[3] = node.order();
        if (fail) throw std::runtime_error("construction-order probe");
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [&] { return started; });
    }
    set_node_order(1);
    seen[4] = runtime_flag_node_order();
    {
        std::lock_guard<std::mutex> lock(mutex);
        changed = true;
    }
    ready.notify_one();
    bool raised = false;
    try {
        async_exec_wait();
    } catch (const std::runtime_error& error) {
        if (!fail || std::string(error.what()) != "construction-order probe") throw;
        raised = true;
    }
    auto caller = runtime_flag_node_order();
    set_node_order(previous);
    if (timed_out) throw std::runtime_error("construction setter waited for the job");
    if (raised != fail || caller != 1)
        throw std::runtime_error("worker leaked its flag or swallowed its exception");
    return seen;
}
}
''', jt.compiler.cc_flags + " " + jt.compiler.opt_flags)

    def _check(self, fail):
        before = jt.flags.node_order
        result = self.probe.check_construction_order(fail)
        if not result:
            self.skipTest("requires two permitted CPUs in one cache domain")
        self.assertEqual(list(result), [0, 0, 2, 1 << 60, 1])
        self.assertEqual(jt.flags.node_order, before)

    def test_worker_and_caller_can_change_their_own_construction_order(self):
        self._check(False)

    def test_worker_exception_preserves_the_caller_and_next_job(self):
        self._check(True)
        self._check(False)
