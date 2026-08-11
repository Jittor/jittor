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
        jt.flags.amp_reg = 2 | 4 | 8 | 16
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.amp_reg = 0
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()
