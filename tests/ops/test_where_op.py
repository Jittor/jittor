import unittest

import jittor as jt

from _helpers.operator_where_cases import WhereOpCases


class TestWhereOp(WhereOpCases, unittest.TestCase):
    __test__ = True


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestWhereOpCuda(WhereOpCases, unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        cls._previous_use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    @classmethod
    def tearDownClass(cls):
        jt.sync_all()
        jt.flags.use_cuda = cls._previous_use_cuda


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestWhereOpCub(TestWhereOpCuda):
    def setUp(self):
        self.where = jt.compile_extern.cub_ops.cub_where


if __name__ == "__main__":
    unittest.main()
