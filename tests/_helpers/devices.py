"""Legacy device base classes shared by operator tests."""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability

import unittest

import jittor as jt


def cuda_test_case(use_cuda=1):
    @_test_preserve_policy(jt, 'use_cuda')
    @unittest.skipIf(not _test_capability.any_accelerator_enabled(backend=jt), "No accelerator backend enabled")
    class TestCudaBase(unittest.TestCase):
        def setUp(self):
            # Remember what was set rather than assuming the process started on
            # CPU. Jittor enables CUDA by default whenever a GPU is present, so
            # a tearDown that hard-codes 0 switches the accelerator off for
            # every test that runs after this class -- across files, since the
            # flag is process-global.
            from contextlib import ExitStack as _TestPolicyStack
            _test_policy_stack = _TestPolicyStack()
            self.addCleanup(_test_policy_stack.close)
            self._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=use_cuda))

        def tearDown(self):
            # Flush first: the flag setter evaluates the pending graph, and it
            # does so under the value being replaced. Draining here keeps that
            # work with the device it was built for.
            from contextlib import ExitStack as _TestPolicyStack
            with _TestPolicyStack() as _test_policy_stack:
                jt.sync_all()
                _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._previous_use_cuda))

    return TestCudaBase
