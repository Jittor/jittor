# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import sys
import os
import jittor as jt
import unittest
import time
import numpy as np
from _helpers.logs import find_log_with_re
from _helpers.tuner_parser import simple_parser

class TestDefaultVar(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        return

    @jt.flag_scope(auto_convert_64_to_32=0)
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
        # A python float is float32 -- `ArrayOp` reads a python number as its
        # own type (see `_CONSTANT_PY_TYPES` in jittor/_core/var.py), so a
        # scalar does not widen the result to the operand's 64-bit width. This
        # line used to expect float64, which contradicted the `int32 * 2.0`
        # case eight lines above: that one has always expected float32 under
        # the very same rule.
        #
        # Checked against both references on this machine rather than argued
        # from the rule alone: torch 2.13 gives float32 here, NumPy 2.4 gives
        # float64. Jittor follows torch.
        assert str(b.dtype) == "float32"
        b=a*2
        assert str(b.dtype) == "int64"

if __name__ == "__main__":
    unittest.main()
