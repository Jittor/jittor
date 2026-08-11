import unittest

import jittor as jt

from _helpers.devices import cuda_test_case
from _helpers.operator_unary_cases import UnaryOpCases


class TestUnaryOp(UnaryOpCases, unittest.TestCase):
    __test__ = True


class TestUnaryOpCuda(UnaryOpCases, cuda_test_case(2)):
    __test__ = True


class TestUnaryOpCpuFp16(UnaryOpCases, cuda_test_case(0)):
    __test__ = True

    def setUp(self):
        jt.flags.amp_reg = 2 | 4 | 8 | 16

    def tearDown(self):
        jt.flags.amp_reg = 0


class TestUnaryOpCudaFp16(UnaryOpCases, cuda_test_case(2)):
    __test__ = True

    def setUp(self):
        super().setUp()
        jt.flags.amp_reg = 2 | 4 | 8 | 16

    def tearDown(self):
        jt.flags.amp_reg = 0
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
