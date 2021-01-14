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

def pool(x, size, op):
    N,H,W,C = x.shape
    h = (H+size-1)//size
    w = (W+size-1)//size
    return x.reindex_reduce(op, [N,h,w,C], [
        "i0", # Nid
        f"i1/{size}", # Hid
        f"i2/{size}", # Wid
        "i3", # Cid
    ])

def pool_naive(x, size, op):
    N,H,W,C = x.shape
    h = (H+size-1)//size
    w = (W+size-1)//size
    y = np.zeros([N,h,w,C], dtype="float64")
    x = np.float64(x)
    if op=="maximum":
        y[:] = -1e100
        fop = lambda x,y: np.maximum(x,y)
    elif op=="minimum": 
        y[:] = 1e100
        fop = lambda x,y: np.minimum(x,y)
    elif op=="multiply":
        y[:] = 1
        fop = lambda x,y: x*y
    else:
        assert op=="add"
        fop = lambda x,y: x+y
    for i0 in range(N):
        for i1 in range(H):
            for i2 in range(W):
                for i3 in range(C):
                    y[i0,i1//size,i2//size,i3] = \
                        fop(y[i0,i1//size,i2//size,i3], x[i0,i1,i2,i3])
    return y
    
ops = ["maximum", "minimum", "multiply", "add"]

class TestReindexReduceOp(unittest.TestCase):
    def test_pool(self):
        N,H,W,C = 3,10,10,4
        size=3
        for op in ops:
            x = jt.random([N,H,W,C])
            y = pool(x, size, op)
            ny = pool_naive(x.data, size, op)
            assert np.allclose(y.data, ny), (op, y.data, ny)
    
    def test_pool_grad(self):
        jt.set_seed(1)
        N,H,W,C = 2,7,7,2
        size=3
        # ops = ["maximum"]
        for op in ops:
            x = jt.random([N,H,W,C])
            y = pool(x, size, op)
            mask = jt.random(y.shape)
            loss = (y*mask).sum()
            dx = jt.grad(loss, x)
            jdx = dx.data
            nx = x.data
            nmask = mask.data
            _, (ndx,) = ngrad(lambda args: (pool_naive(args[0], size, op)*nmask).sum(), [nx], 1e-6)
            assert np.allclose(jdx, ndx), (op, jdx[0,:,:,0], ndx[0,:,:,0])
        
    def test_error(self):
        jt.random([3]).reindex_reduce("add", [3], ["i0"])
        expect_error(lambda: jt.random([3]).reindex_reduce("add", [3], []))
        expect_error(lambda: jt.random([3]).reindex_reduce("add", [3], ["i0","i0"]))
        expect_error(lambda: jt.random([3]).reindex_reduce("???", [3], ["i0"]))
        expect_error(lambda: jt.random([3]).reindex_reduce("add", [-1], ["i0"]))

@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestReindexReduceOpCuda(TestReindexReduceOp):
    def setUp(self):
        # TODO: replace to 2
        jt.flags.use_cuda = 1
    def tearDown(self):
        jt.flags.use_cuda = 0

if __name__ == "__main__":
    unittest.main()