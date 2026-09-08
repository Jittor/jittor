
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
import unittest

import jittor as jt

from _helpers.operator_reindex_cases import ReindexOpCases


class TestReindexOp(ReindexOpCases, unittest.TestCase):
    __test__ = True


@_test_preserve_policy(jt, 'use_cuda')
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestReindexOpCuda(ReindexOpCases, unittest.TestCase):
    __test__ = True

    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._previous_use_cuda))


if __name__ == "__main__":
    unittest.main()
