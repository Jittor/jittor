# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import os

class TestProfiler(unittest.TestCase):
    def test_profiler(self):
        a = jt.rand(1000,1000)
        b = jt.rand(1000,1000)
        jt.sync_all()
        with jt.profile_scope(10, 100, profiler_record_peek=1) as rep:
            jt.matmul(a, b).sync()
        x = float(rep[-1][4])
        y = float(rep[-2][4])
        assert abs(x-y)/x < 1e-3

if __name__ == "__main__":
    unittest.main()