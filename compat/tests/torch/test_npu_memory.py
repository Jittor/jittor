"""Real ACL allocator/driver checks; host mocks are a separate contract file."""
from contextlib import ExitStack
from functools import lru_cache
import gc
import unittest

import jittor as jt
import torch


@lru_cache(maxsize=1)
def _workspace_probe():
    # Compile only when an actual ACL test requests this helper. It allocates
    # on the real physical device through the production Temp workspace pool;
    # no mocked driver or relabelled host storage is involved.
    import jittor_utils
    return jittor_utils.compile_module(r'''
#include "mem/allocator.h"
#include "mem/mem_info.h"
#include "runtime/device.h"
#include <string>
namespace jittor {
// @pyjt(temp_workspace_peak)
vector<int64> temp_workspace_peak(int device, int64 bytes) {
    auto* pool = get_allocator({accelerator_backend_id(), device}, true);
    CHECK(pool->is_cuda() && pool->device() == device);
    CHECK(std::string(pool->name()).find("temp") != std::string::npos)
        << "This test requires the production Temp workspace pool";
    gc_device_cache(device);
    reset_device_memory_peaks(device);
    auto before = device_pool_memory_stats(device);
    {
        Allocation workspace(pool, size_t(bytes));
        CHECK(workspace.ptr);
        // Deliberately do not sample statistics while the workspace is live.
    }
    auto released = device_pool_memory_stats(device);
    gc_device_cache(device);
    auto collected = device_pool_memory_stats(device);
    before.insert(before.end(), released.begin(), released.end());
    before.insert(before.end(), collected.begin(), collected.end());
    return before;
}
}''', jt.compiler.cc_flags)


class TestNPUMemory(unittest.TestCase):
    def setUp(self):
        if not getattr(jt.compiler, "has_acl", False):
            self.skipTest("requires the ACL backend and real NPU")
        from jittor.compat.torch.publication import namespace_owner
        from jittor._runtime.fallback import forbid_backend_fallbacks
        self.assertIs(namespace_owner(torch), jt)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(forbid_backend_fallbacks())
        self.fallback_before = jt.core.backend_fallback_count()
        self.addCleanup(self.check_fallback)
        self.device = jt.core.current_device()

    def check_fallback(self):
        jt.sync_all(True)
        self.assertEqual(jt.core.backend_fallback_count(), self.fallback_before)

    def test_actual_allocation_and_transient_peak_survive_free(self):
        gc.collect()
        torch.npu.empty_cache()
        before = torch.npu.memory_allocated(self.device)
        torch.npu.reset_peak_memory_stats(self.device)
        count = 2 * 1024 * 1024
        value = torch.empty(count, dtype=torch.float32, device="npu:%d" % self.device)
        value.sync()
        self.assertEqual(value.location(), "device")
        self.assertEqual(value.get_device(), self.device)
        during = torch.npu.memory_allocated(self.device)
        self.assertGreaterEqual(during - before, count * 4)
        self.assertGreaterEqual(torch.npu.memory_reserved(self.device), during)
        del value
        gc.collect()
        torch.npu.empty_cache()
        after = torch.npu.memory_allocated(self.device)
        self.assertLess(after, during)
        self.assertGreaterEqual(torch.npu.max_memory_allocated(self.device), during)
        self.assertGreaterEqual(torch.npu.max_memory_reserved(self.device),
                                torch.npu.max_memory_allocated(self.device))
        torch.npu.reset_peak_memory_stats(self.device)
        self.assertEqual(torch.npu.max_memory_allocated(self.device),
                         torch.npu.memory_allocated(self.device))
        self.assertEqual(torch.npu.max_memory_reserved(self.device),
                         torch.npu.memory_reserved(self.device))

    def test_peak_without_intermediate_python_memory_query(self):
        torch.npu.reset_peak_memory_stats(self.device)
        before = torch.npu.memory_allocated(self.device)
        value = torch.empty(2 * 1024 * 1024, device="npu:%d" % self.device, dtype=torch.float32)
        value.sync()
        del value
        gc.collect()
        torch.npu.empty_cache()
        self.assertGreaterEqual(torch.npu.max_memory_allocated(self.device), before + 8 * 1024**2)

    def test_driver_info_and_cpu_default_are_preserved(self):
        with jt.runtime.scope(use_cuda=0):
            before = jt.core.current_device()
            for index in range(jt.core.get_device_count()):
                free, total = torch.npu.mem_get_info(index)
                self.assertGreater(total, 0)
                self.assertGreaterEqual(free, 0)
                self.assertLessEqual(free, total)
                self.assertEqual(jt.core.current_device(), before)
                self.assertGreaterEqual(torch.npu.memory_reserved(index),
                                        torch.npu.memory_allocated(index))
            cpu = torch.empty(8)
            cpu.sync()
            self.assertEqual(cpu.location(), "cpu")

    def test_cpu_allocations_and_cache_are_isolated_from_npu(self):
        jt.sync_all(True)
        torch.npu.empty_cache()
        before = list(jt.core.device_pool_memory_stats(self.device))
        with jt.runtime.scope(use_cuda=0):
            cpu = torch.empty(4 * 1024 * 1024, device="cpu", dtype=torch.float32)
            cpu.sync()
            self.assertEqual(cpu.location(), "cpu")
            self.assertEqual(list(jt.core.device_pool_memory_stats(self.device)), before)
            del cpu
            gc.collect()
            host_cached = jt.core.device_memory_reserved(-1)
            self.assertGreaterEqual(host_cached, 16 * 1024**2)
            torch.npu.empty_cache()
            self.assertEqual(jt.core.device_memory_reserved(-1), host_cached)
            self.assertEqual(list(jt.core.device_pool_memory_stats(self.device)), before)
            self.assertEqual(jt.runtime.use_cuda, 0)

    def test_empty_cache_preserves_live_npu_storage(self):
        value = torch.tensor([1., 2., 3.], device="npu:%d" % self.device)
        value.sync()
        before = torch.npu.memory_allocated(self.device)
        torch.npu.empty_cache()
        self.assertEqual(torch.npu.memory_allocated(self.device), before)
        self.assertEqual(value.location(), "device")
        self.assertEqual(value.get_device(), self.device)
        self.assertEqual(value.tolist(), [1., 2., 3.])

    def test_real_temp_workspace_transient_peak(self):
        jt.sync_all(True)
        size = 8 * 1024**2
        stats = list(_workspace_probe().temp_workspace_peak(self.device, size))
        before, released, collected = stats[:4], stats[4:8], stats[8:12]
        self.assertEqual(len(stats), 12)
        self.assertEqual(released[0], before[0])
        self.assertGreaterEqual(released[2], before[0] + size)
        self.assertGreaterEqual(released[1], before[1] + size)
        self.assertGreaterEqual(released[3], released[1])
        self.assertEqual(collected[0], before[0])
        self.assertLess(collected[1], released[1])
        self.assertEqual(collected[2:], released[2:])


if __name__ == "__main__":
    unittest.main()
