
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
import unittest

import jittor as jt

from _helpers.operator_reduce_cases import ReduceOpCases


class TestReduceOp(ReduceOpCases, unittest.TestCase):
    __test__ = True


class TestReduceOp2(ReduceOpCases, unittest.TestCase):
    __test__ = True
    keepdims = True


@_test_preserve_policy(jt, 'use_cuda')
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestReduceOpCuda(ReduceOpCases, unittest.TestCase):
    __test__ = True

    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=2))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._previous_use_cuda))


@_test_preserve_policy(jt, 'use_cuda')
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestReduceOpCuda2(ReduceOpCases, unittest.TestCase):
    __test__ = True
    keepdims = True

    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=2))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._previous_use_cuda))


class TestReduceOpMisc(unittest.TestCase):
    def test_negtive_dim(self):
        a = jt.array([[1, 2], [3, 4]])
        assert (a.sum(-1).data == [3, 7]).all()
        assert (a.sum(-2).data == [4, 6]).all()


if __name__ == "__main__":
    unittest.main()
