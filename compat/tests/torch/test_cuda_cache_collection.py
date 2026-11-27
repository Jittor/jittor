"""Real CUDA cache collection must release idle blocks without touching owners."""
import gc
import unittest

import numpy as np
import jittor as jt
import torch

from _helpers.capability import require_accelerator
from jittor._runtime import forbid_backend_fallbacks
from jittor.compat.torch.context import get_install_context


class TestCudaCacheCollection(unittest.TestCase):
    def setUp(self):
        require_accelerator('cuda')
        self.runtime = get_install_context(jt).state['cuda_runtime']
        self.previous_mode = self.runtime.empty_cache_mode
        self.runtime.empty_cache_mode = 'gc'
        self.addCleanup(setattr, self.runtime, 'empty_cache_mode', self.previous_mode)

    def _released_block(self):
        tensor = torch.ones(8 * 1024 * 1024, dtype=torch.float32, device='cuda')
        tensor.sync()
        device = int(tensor.device_id)
        reserved = int(jt.core.device_memory_reserved(device))
        del tensor
        gc.collect()
        jt.sync_all(True)
        return device, reserved

    def test_empty_cache_releases_unused_blocks_and_keeps_live_data(self):
        with forbid_backend_fallbacks():
            live = torch.tensor([1., 3., -2.], device='cuda')
            live.sync()
            device, reserved = self._released_block()
            torch.cuda.empty_cache()
            self.assertLess(jt.core.device_memory_reserved(device), reserved)
            self.assertEqual(live.location(), 'device')
            np.testing.assert_array_equal(live.detach().cpu().numpy(), [1., 3., -2.])

    def test_per_device_collection_uses_real_owner_and_rejects_bad_ordinals(self):
        with forbid_backend_fallbacks():
            live = torch.tensor([7., 8.], device='cuda')
            live.sync()
            device, reserved = self._released_block()
            jt.core.gc_device_cache(device)
            self.assertLess(jt.core.device_memory_reserved(device), reserved)
            self.assertEqual(int(live.device_id), device)
            self.assertEqual(live.location(), 'device')
            np.testing.assert_array_equal(live.detach().cpu().numpy(), [7., 8.])
            for invalid in (-1, torch.cuda.device_count()):
                with self.assertRaises(jt.JittorError):
                    jt.core.gc_device_cache(invalid)

    def test_per_device_collection_does_not_collect_the_other_gpu(self):
        if torch.cuda.device_count() < 2:
            self.skipTest('two real CUDA devices are required')
        with forbid_backend_fallbacks():
            live = [torch.tensor([index + 1., index + 3.], device='cuda:%d' % index)
                    for index in (0, 1)]
            for value in live:
                value.sync()
            idle = [torch.ones(8 * 1024 * 1024, dtype=torch.float32,
                               device='cuda:%d' % index) for index in (0, 1)]
            for value in idle:
                value.sync()
            # The loop's local must not retain GPU 1's nominally idle block.
            del value
            del idle
            gc.collect()
            jt.sync_all(True)
            before = [int(jt.core.device_memory_reserved(index)) for index in (0, 1)]
            # Deliberately select a different ambient GPU: collection follows
            # the requested pool's real owner rather than this thread's flag.
            with torch.cuda.device(1):
                jt.core.gc_device_cache(0)
            after_zero = [int(jt.core.device_memory_reserved(index)) for index in (0, 1)]
            self.assertLess(after_zero[0], before[0])
            self.assertEqual(after_zero[1], before[1])
            for index, value in enumerate(live):
                self.assertEqual(int(value.device_id), index)
                self.assertEqual(value.location(), 'device')
                np.testing.assert_array_equal(value.detach().cpu().numpy(), [index + 1., index + 3.])
            before_one = int(jt.core.device_memory_reserved(1))
            with torch.cuda.device(0):
                jt.core.gc_device_cache(1)
            self.assertLess(jt.core.device_memory_reserved(1), before_one)
            for index, value in enumerate(live):
                self.assertEqual(int(value.device_id), index)
                self.assertEqual(value.location(), 'device')
                np.testing.assert_array_equal(value.detach().cpu().numpy(), [index + 1., index + 3.])

    def test_fetch_staging_collection_preserves_live_device_and_host_results(self):
        with forbid_backend_fallbacks():
            live = torch.ones(1024 * 1024, device='cuda')
            # The native read uses fetch staging; keep both actual results
            # alive while collecting idle static staging blocks repeatedly.
            host = live.detach().numpy()
            for _ in range(3):
                jt.sync_all(True)
                jt.gc()
            self.assertEqual(live.location(), 'device')
            np.testing.assert_array_equal(host, np.ones(1024 * 1024, dtype=np.float32))
            self.assertEqual(live.sum().item(), 1024 * 1024)


if __name__ == '__main__':
    unittest.main()
