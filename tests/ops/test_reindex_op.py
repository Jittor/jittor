import unittest

import jittor as jt

from _helpers.operator_reindex_cases import ReindexOpCases


class TestReindexOp(ReindexOpCases, unittest.TestCase):
    __test__ = True


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestReindexOpCuda(ReindexOpCases, unittest.TestCase):
    __test__ = True

    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()
