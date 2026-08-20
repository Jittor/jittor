import unittest

import jittor as jt

from _helpers.devices import cuda_test_case
from _helpers.operator_binary_cases import BinaryOpCases


class TestBinaryOp(BinaryOpCases, unittest.TestCase):
    __test__ = True


class TestBinaryOpCuda(BinaryOpCases, cuda_test_case(2)):
    __test__ = True


class TestBinaryOpCpuFp16(BinaryOpCases, unittest.TestCase):
    __test__ = True

    def setUp(self):
        jt.flags.amp_reg = 2 | 4 | 8 | 16

    def tearDown(self):
        jt.flags.amp_reg = 0


@unittest.skipIf(not jt.has_cuda, "no cuda found")
class TestBinaryOpCudaFp16(BinaryOpCases, unittest.TestCase):
    __test__ = True

    def setUp(self):
        # Restore what was there rather than assuming 0. On a machine with a
        # GPU the default is 1, so hard-coding 0 here switches the accelerator
        # off for every test that runs after this class.
        self._use_cuda = jt.flags.use_cuda
        self._amp_reg = jt.flags.amp_reg
        jt.flags.amp_reg = 2 | 4 | 8 | 16
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.amp_reg = self._amp_reg
        jt.flags.use_cuda = self._use_cuda


if __name__ == "__main__":
    unittest.main()
