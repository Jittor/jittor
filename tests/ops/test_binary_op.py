
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
import unittest

import jittor as jt

from _helpers.devices import cuda_test_case
from _helpers.operator_binary_cases import BinaryOpCases


class TestBinaryOp(BinaryOpCases, unittest.TestCase):
    __test__ = True


class TestBinaryOpCuda(BinaryOpCases, cuda_test_case(2)):
    __test__ = True


@_test_preserve_policy(jt, 'amp_reg')
class TestBinaryOpCpuFp16(BinaryOpCases, unittest.TestCase):
    __test__ = True

    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        _test_policy_stack.enter_context(jt.runtime.scope(amp_reg=2 | 4 | 8 | 16))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(amp_reg=0))


@_test_preserve_policy(jt, 'amp_reg', 'use_cuda')
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "no cuda found")
class TestBinaryOpCudaFp16(BinaryOpCases, unittest.TestCase):
    __test__ = True

    def setUp(self):
        # Restore what was there rather than assuming 0. On a machine with a
        # GPU the default is 1, so hard-coding 0 here switches the accelerator
        # off for every test that runs after this class.
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._use_cuda = jt.introspection.policy.runtime.use_cuda
        self._amp_reg = jt.introspection.policy.runtime.amp_reg
        _test_policy_stack.enter_context(jt.runtime.scope(amp_reg=2 | 4 | 8 | 16))
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(amp_reg=self._amp_reg))
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._use_cuda))


if __name__ == "__main__":
    unittest.main()
