
from _helpers import capability as _test_capability
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
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

@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
class TestFetcherCuda(TestFetcher):
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