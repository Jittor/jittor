# ***************************************************************
# Copyright (c) 2019 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
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
if __name__ == "__main__":
    unittest.main()
