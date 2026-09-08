
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
import unittest

import jittor as jt

from _helpers.devices import cuda_test_case
from _helpers.operator_unary_cases import UnaryOpCases


class TestUnaryOp(UnaryOpCases, unittest.TestCase):
    __test__ = True


class TestUnaryOpCuda(UnaryOpCases, cuda_test_case(2)):
    __test__ = True


@_test_preserve_policy(jt, 'amp_reg')
class TestUnaryOpCpuFp16(UnaryOpCases, cuda_test_case(0)):
    __test__ = True

    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        super().setUp()
        self._previous_amp_reg = jt.introspection.policy.runtime.amp_reg
        _test_policy_stack.enter_context(jt.runtime.scope(amp_reg=2 | 4 | 8 | 16))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(amp_reg=self._previous_amp_reg))
            super().tearDown()


@_test_preserve_policy(jt, 'amp_reg')
class TestUnaryOpCudaFp16(UnaryOpCases, cuda_test_case(2)):
    __test__ = True

    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        super().setUp()
        self._previous_amp_reg = jt.introspection.policy.runtime.amp_reg
        _test_policy_stack.enter_context(jt.runtime.scope(amp_reg=2 | 4 | 8 | 16))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(amp_reg=self._previous_amp_reg))
            super().tearDown()


if __name__ == "__main__":
    unittest.main()
