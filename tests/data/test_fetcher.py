
from _helpers import capability as _test_capability
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import time
import jittor as jt
import numpy as np
from jittor import compile_extern

class TestFetcher(unittest.TestCase):
    def test_fetch(self):
        a = jt.array([1,2,3])
        a = a*2
        v = []
        jt.fetch(a, lambda a: v.append(a))
        jt.fetch(1, 2, 3, a, 
            lambda x, y, z, a: self.assertTrue(x==1 and y==2 and z==3 and isinstance(a, np.ndarray))
        )
        jt.sync_all(True)
        assert len(v)==1 and (v[0]==[2,4,6]).all()

    def test_poll_fetches_leaves_unrelated_work_pending(self):
        with jt.flag_scope(auto_graph_replay=0, backend_fallback="error"):
            source = jt.arange(32).cast("int32").sync()
            requested = source * 3
            unrelated = jt.ones((113,), dtype="float32") * 17
            assert requested.location() == unrelated.location() == "none"
            arrays = []
            jt.fetch(requested, lambda array: arrays.append(np.array(array, copy=True)))
            deadline = time.monotonic() + 5
            try:
                while not arrays and time.monotonic() < deadline:
                    jt.poll_fetches()
                    if not arrays:
                        time.sleep(0.0001)
                assert len(arrays) == 1
                np.testing.assert_array_equal(arrays[0], np.arange(32, dtype=np.int32) * 3)
                assert requested.location() == ("device" if jt.flags.use_cuda else "cpu")
                assert unrelated.location() == "none"
                jt.poll_fetches()
                assert len(arrays) == 1
            finally:
                jt.sync_all(True)

@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
class TestFetcherCuda(TestFetcher):
    def test_fetch_waits_once_per_producer(self):
        with jt.flag_scope(use_cuda=1, device_id=0, auto_graph_replay=0,
                           backend_fallback="error"):
            expected = [np.arange(48, dtype=np.float32) + i for i in range(5)]
            values = [jt.array(item) for item in expected]
            jt.sync(values, True, False)
            self.assertTrue(all(value.location() == "device" for value in values))
            before = jt.core._cuda_stream_dependency_count(0, 0)
            got = []
            jt.fetch(*values, lambda *arrays: got.extend(array.copy() for array in arrays))
            jt.sync_all(True)
            after = jt.core._cuda_stream_dependency_count(0, 0)
            self.assertEqual(after - before, 1)
            self.assertEqual(len(got), len(expected))
            for actual, reference in zip(got, expected):
                np.testing.assert_array_equal(actual, reference)

    @classmethod
    def setUpClass(self):
        from _helpers.runtime_policy import fixture_stack
        _test_policy_stack = fixture_stack(self, class_scope=True)
        try:
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))
        except BaseException:
            _test_policy_stack.close()
            raise

    @classmethod
    def tearDownClass(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=0))

if __name__ == "__main__":
    unittest.main()
