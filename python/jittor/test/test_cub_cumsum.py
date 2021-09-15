# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor import compile_extern
if jt.has_cuda:
    from jittor.compile_extern import cublas_ops, cudnn_ops, cub_ops
else:
    cublas_ops = cudnn_ops = cub_ops = None


def test_forward(shape, dim=None):
    x = jt.random(shape)
    y = jt.numpy_cumsum(x)
    y_ = jt.cub_cumsum(x)
    assert(np.allclose(y.data, y_.data))

def test_backward(shape, dim=None):
    x = jt.random(shape)
    z = jt.random(shape)

    y = jt.numpy_cumsum(x)
    loss = (y * z).sum()
    grad = jt.grad(loss, x)
    
    y_ = jt.cub_cumsum(x)
    loss_ = (y_ * z).sum()
    grad_ = jt.grad(loss_, x)
    assert(np.allclose(grad.data, grad_.data))

class TestCubCumsumOp(unittest.TestCase):
    @unittest.skipIf(cub_ops==None, "Not use cub, Skip")
    @jt.flag_scope(use_cuda=1)
    def test_1d(self):
        test_forward([20])
        test_forward([3007])
        test_forward([3007], 0)
        test_forward([3007], -1)

    @unittest.skipIf(cub_ops==None, "Not use cub, Skip")
    @jt.flag_scope(use_cuda=1)
    def test_1d_backward(self):
        test_backward([20])
        test_backward([3007])
        test_backward([3007], 0)
        test_backward([3007], -1)

    @unittest.skipIf(cub_ops==None, "Not use cub, Skip")
    @jt.flag_scope(use_cuda=1)
    def test_2d(self):
        test_forward([5,5])
        test_forward([2000, 3007])
        test_forward([2000, 3007], 1)
        test_forward([2000, 3007], -1)

    @unittest.skipIf(cub_ops==None, "Not use cub, Skip")
    @jt.flag_scope(use_cuda=1)
    def test_2d_backward(self):
        test_backward([5,5])
        test_backward([2000, 3007])
        test_backward([2000, 3007], 1)
        test_backward([2000, 3007], -1)

    @unittest.skipIf(cub_ops==None, "Not use cub, Skip")
    @jt.flag_scope(use_cuda=1)
    def test_nd(self):
        test_forward([5,6,7,8], 0)
        test_forward([5,6,7,8], 1)
        test_forward([5,6,7,8], 2)
        test_forward([5,6,7,8], 3)
        test_forward([5,6,7,8], -1)
        test_forward([16,14,14,2048], 0)
        test_forward([16,14,14,2048], 1)
        test_forward([16,14,14,2048], 2)
        test_forward([16,14,14,2048], 3)
        test_forward([16,14,14,2048], -1)
        test_forward([16,14,14,2047], 3)

    @unittest.skipIf(cub_ops==None, "Not use cub, Skip")
    @jt.flag_scope(use_cuda=1)
    def test_nd_backward(self):
        test_backward([5,6,7,8], 0)
        test_backward([5,6,7,8], 1)
        test_backward([5,6,7,8], 2)
        test_backward([5,6,7,8], 3)
        test_backward([5,6,7,8], -1)
        test_backward([16,14,14,2048], 0)
        test_backward([16,14,14,2048], 1)
        test_backward([16,14,14,2048], 2)
        test_backward([16,14,14,2048], 3)
        test_backward([16,14,14,2048], -1)
        test_backward([16,14,14,2047], 3)

if __name__ == "__main__":
    unittest.main()