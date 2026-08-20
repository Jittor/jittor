"""Legacy device base classes shared by operator tests."""

import unittest

import jittor as jt


def cuda_test_case(use_cuda=1):
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    class TestCudaBase(unittest.TestCase):
        def setUp(self):
            # Remember what was set rather than assuming the process started on
            # CPU. Jittor enables CUDA by default whenever a GPU is present, so
            # a tearDown that hard-codes 0 switches the accelerator off for
            # every test that runs after this class -- across files, since the
            # flag is process-global.
            self._previous_use_cuda = jt.flags.use_cuda
            jt.flags.use_cuda = use_cuda

        def tearDown(self):
            # Flush first: the flag setter evaluates the pending graph, and it
            # does so under the value being replaced. Draining here keeps that
            # work with the device it was built for.
            jt.sync_all()
            jt.flags.use_cuda = self._previous_use_cuda

    return TestCudaBase
