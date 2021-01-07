# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#   Meng-Hao Guo <guomenghao1997@gmail.com>
#   Dun Liang <randonlang@gmail.com>.
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn
import unittest
import numpy as np

class TestBMM(unittest.TestCase):
    @unittest.skipIf(not jt.has_cuda, "No cuda found")
    def test_bmm_cuda(self):
        def check(batch, n, m, k):
            def calc(use_cuda, a, b, mask):
                jt.flags.use_cuda = use_cuda
                a = jt.array(a)
                b = jt.array(b)
                mask = jt.array(mask)
                c = nn.bmm(a, b)
                da, db = jt.grad(c*mask, [a, b])
                return c.data, da.data, db.data
            mask = np.random.rand(batch, n, k).astype("float32")
            a = np.random.rand(batch, n, m).astype("float32")
            b = np.random.rand(batch, m, k).astype("float32")
            a1,a2,a3 = calc(0, a, b, mask)
            b1,b2,b3 = calc(1, a, b, mask)
            assert np.allclose(a1, b1)
            assert np.allclose(a2, b2)
            assert np.allclose(a3, b3)
        check(10,3,4,5)
        check(10,8,8,8)
        check(10,8,1,8)
        check(10,8,8,1)
        check(10,1,8,8)
        check(1,7,8,8)


if __name__ == "__main__":
    unittest.main()