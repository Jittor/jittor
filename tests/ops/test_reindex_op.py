import unittest

import jittor as jt

from _helpers.operator_reindex_cases import ReindexOpCases


class TestReindexOp(ReindexOpCases, unittest.TestCase):
    __test__ = True


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestReindexOpCuda(ReindexOpCases, unittest.TestCase):
    __test__ = True

    def setUp(self):
        self._previous_use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.sync_all()
        jt.flags.use_cuda = self._previous_use_cuda


if __name__ == "__main__":
    unittest.main()
