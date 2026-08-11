import unittest

import jittor as jt

from _helpers.operator_reduce_cases import ReduceOpCases


class TestReduceOp(ReduceOpCases, unittest.TestCase):
    __test__ = True


class TestReduceOp2(ReduceOpCases, unittest.TestCase):
    __test__ = True
    keepdims = True


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestReduceOpCuda(ReduceOpCases, unittest.TestCase):
    __test__ = True

    def setUp(self):
        jt.flags.use_cuda = 2

    def tearDown(self):
        jt.flags.use_cuda = 0


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestReduceOpCuda2(ReduceOpCases, unittest.TestCase):
    __test__ = True
    keepdims = True

    def setUp(self):
        jt.flags.use_cuda = 2

    def tearDown(self):
        jt.flags.use_cuda = 0


class TestReduceOpMisc(unittest.TestCase):
    def test_negtive_dim(self):
        a = jt.array([[1, 2], [3, 4]])
        assert (a.sum(-1).data == [3, 7]).all()
        assert (a.sum(-2).data == [4, 6]).all()


if __name__ == "__main__":
    unittest.main()
