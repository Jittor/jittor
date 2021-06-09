# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error
from .test_grad import ngrad
from .test_cuda import test_cuda

class TestTernaryOp(unittest.TestCase):
    def test_with_np(self):
        np.random.seed(0)
        a = np.random.rand(5,10).astype("float32")
        b = np.random.rand(5,10).astype("float32")
        ja = jt.array(a)
        jb = jt.array(b)
        jc = jt.ternary(ja>jb, ja, jb)
        assert (jc.data==np.maximum(a,b)).all(), f"\n{jc.data}\n{np.maximum(a,b)}\n{a}\n{b}"
        jda, jdb = jt.grad(jc, [ja, jb])
        assert (jda.data==(a>b)*1).all()
        assert (jdb.data==1-(a>b)).all()

    def test_min(self):
        np.random.seed(1)
        a = np.random.rand(5,10).astype("float32")
        b = np.random.rand(5,10).astype("float32")
        ja = jt.array(a)
        jb = jt.array(b)
        jc = jt.minimum(ja,jb)
        assert (jc.data==np.minimum(a,b)).all(), f"\n{jc.data}\n{np.minimum(a,b)}\n{a}\n{b}"
        jda, jdb = jt.grad(jc, [ja, jb])
        assert (jda.data==(a<b)*1).all()
        assert (jdb.data==1-(a<b)).all()

class TestTernaryOpCuda(TestTernaryOp, test_cuda(2)):
    pass

if __name__ == "__main__":
    unittest.main()