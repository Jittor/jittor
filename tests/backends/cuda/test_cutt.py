# ***************************************************************
# Copyright (c) 2019 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
import unittest
import unittest.mock
import jittor as jt
import numpy as np
from jittor import compile_extern
from _helpers.cutt import require_cutt_ops
from _helpers.logs import find_log_with_re
import copy

class TestCutt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cutt_ops = require_cutt_ops()

    @jt.flag_scope(use_cuda=1)
    def test(self):
        t = self.cutt_ops.cutt_test("213")
        assert t.data == 123


class TestCuttUnavailableIsNotSkipped(unittest.TestCase):
    """A failed cuTT build must fail, not skip.

    This is how a broken cuTT build stayed invisible: the transpose ops fall
    back to the built-in kernel and report "cutt is unavailable", so the whole
    cuTT suite skipped and the gate stayed green.
    """

    def test_load_failure_with_cuda_and_cutt_enabled_is_an_error(self):
        if not jt.has_cuda:
            self.skipTest("no CUDA on this machine")
        import _helpers.cutt as helper
        original = helper.get_library_ops
        helper.get_library_ops = lambda name, load=False: None
        raised = None
        try:
            with unittest.mock.patch.dict(os.environ, {"use_cutt": "1"}):
                try:
                    helper.require_cutt_ops()
                except BaseException as exc:  # SkipTest is not an Exception
                    raised = exc
        finally:
            helper.get_library_ops = original
        # Catching BaseException rather than asserting on AssertionError is the
        # point: if the helper goes back to skipping, a SkipTest escaping an
        # assertRaises block would mark this test skipped, which is the very
        # outcome being ruled out.
        self.assertIsNotNone(raised, "a failed cuTT load raised nothing")
        self.assertNotIsInstance(
            raised, unittest.SkipTest,
            "a failed cuTT build was turned into a skip: {}".format(raised))
        self.assertIsInstance(raised, AssertionError)
        self.assertIn("build failed", str(raised))

    def test_disabled_by_env_still_skips(self):
        if not jt.has_cuda:
            self.skipTest("no CUDA on this machine")
        import _helpers.cutt as helper
        with unittest.mock.patch.dict(os.environ, {"use_cutt": "0"}):
            with self.assertRaises(unittest.SkipTest):
                helper.require_cutt_ops()


if __name__ == "__main__":
    unittest.main()
