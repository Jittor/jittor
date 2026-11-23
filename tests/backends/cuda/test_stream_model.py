# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Per-device copy/communication streams and their event dependencies."""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _device_count():
    return int(_test_capability.device_count('cuda', backend=jt))


class TestCudaStreamModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _test_capability.check_accelerator('cuda', backend=jt).enabled or _device_count() < 2:
            raise unittest.SkipTest("two CUDA devices are required")

    def test_copy_and_communication_streams_are_per_device(self):
        handles = {
            (kind, device): jt.core._cuda_stream_handle(kind, device)
            for kind in (0, 1) for device in (0, 1)
        }
        self.assertTrue(all(handles.values()))
        self.assertEqual(len(set(handles.values())), 4)

    def test_h2d_then_compute_and_mixed_device_fetch(self):
        expected = np.arange(4096, dtype="float32")
        values = []
        with jt.flag_scope(use_cuda=1, device_id=0):
            values.append((jt.array(expected) * 3 + 1).sqr())
            values.append((jt.array(expected) * 3 + 3).sqr())
        with jt.flag_scope(use_cuda=1, device_id=1):
            values.append((jt.array(expected) * 5 + 2).sqr())
            values.append((jt.array(expected) * 5 + 4).sqr())

        got = []
        before = jt.core._cuda_stream_dependency_count(0, 0)
        jt.fetch(*values, lambda *arrays: got.extend(v.copy() for v in arrays))
        for _ in range(3):
            jt.sync_all(True)
            if got:
                break
            jt.poll_fetches()
        self.assertEqual(jt.core._cuda_stream_dependency_count(0, 0) - before, 2)
        self.assertEqual(len(got), 4)
        np.testing.assert_array_equal(got[0], (expected * 3 + 1) ** 2)
        np.testing.assert_array_equal(got[1], (expected * 3 + 3) ** 2)
        np.testing.assert_array_equal(got[2], (expected * 5 + 2) ** 2)
        np.testing.assert_array_equal(got[3], (expected * 5 + 4) ** 2)


if __name__ == "__main__":
    unittest.main()
