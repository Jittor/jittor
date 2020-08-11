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
from .test_log import find_log_with_re
import copy
if jt.has_cuda:
    from jittor.compile_extern import cutt_ops
else:
    cutt_ops = None

class TestCutt(unittest.TestCase):
    @unittest.skipIf(cutt_ops==None, "Not use cutt, Skip")
    @jt.flag_scope(use_cuda=1)
    def test(self):
        t = cutt_ops.cutt_test("213")
        assert t.data == 123
if __name__ == "__main__":
    unittest.main()