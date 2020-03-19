# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import sys
import os
import jittor as jt
import unittest
import time
import numpy as np
from .test_reorder_tuner import simple_parser
from .test_log import find_log_with_re

class TestDefaultVar(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        return

    def test_default_var(self):
        a=jt.array((2,3,3), np.float32)
        b=a*2.0
        assert str(b.dtype) == "float32"
        b=a*2
        assert str(b.dtype) == "float32"
        a=jt.array((2,3,3), np.int32)
        b=a*2.0
        assert str(b.dtype) == "float32"
        b=a*2
        assert str(b.dtype) == "int32"

        a=jt.array((2,3,3), np.float64)
        b=a*2.0
        assert str(b.dtype) == "float64"
        b=a*2
        assert str(b.dtype) == "float64"
        a=jt.array((2,3,3), np.int64)
        b=a*2.0
        assert str(b.dtype) == "float64"
        b=a*2
        assert str(b.dtype) == "int64"

if __name__ == "__main__":
    unittest.main()
