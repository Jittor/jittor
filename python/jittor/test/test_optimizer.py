# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#    Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor import nn

class TestOptimizer(unittest.TestCase):
    def test_param_groups(self):
        pa = jt.ones((1,))
        pb = jt.ones((1,))
        data = jt.ones((1,))
        opt = nn.SGD([
            {"params":[pa], "lr":0.1}, 
            {"params":[pb]}, 
        ], 1)
        opt.step(pa*data+pb*data)
        assert pa.data == 0.9 and pb.data == 0, (pa, pb)


if __name__ == "__main__":
    unittest.main()