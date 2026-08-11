"""Legacy device base classes shared by operator tests."""

import unittest

import jittor as jt


def cuda_test_case(use_cuda=1):
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    class TestCudaBase(unittest.TestCase):
        def setUp(self):
            jt.flags.use_cuda = use_cuda

        def tearDown(self):
            jt.flags.use_cuda = 0

    return TestCudaBase
