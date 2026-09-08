
from _helpers import capability as _test_capability
import unittest

import jittor as jt

from _helpers.operator_where_cases import WhereOpCases


class TestWhereOp(WhereOpCases, unittest.TestCase):
    __test__ = True


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestWhereOpCuda(WhereOpCases, unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        from _helpers.runtime_policy import fixture_stack
        _test_policy_stack = fixture_stack(cls, class_scope=True)
        try:
            cls._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))
        except BaseException:
            _test_policy_stack.close()
            raise

    @classmethod
    def tearDownClass(cls):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=cls._previous_use_cuda))


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestWhereOpCub(TestWhereOpCuda):
    def setUp(self):
        self.where = jt.compile_extern.cub_ops.cub_where


if __name__ == "__main__":
    unittest.main()
