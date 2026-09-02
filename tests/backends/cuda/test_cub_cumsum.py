# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
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


def check_forward(shape, dim=None):
    x = jt.random(shape)
    y = jt.numpy_cumsum(x, dim=dim)
    y_ = jt.cub_cumsum(x, dim=dim)
    assert(np.allclose(y.data, y_.data))

def check_backward(shape, dim=None):
    x = jt.random(shape)
    z = jt.random(shape)

    y = jt.numpy_cumsum(x, dim=dim)
    loss = (y * z).sum()
    grad = jt.grad(loss, x)
    
    y_ = jt.cub_cumsum(x, dim=dim)
    loss_ = (y_ * z).sum()
    grad_ = jt.grad(loss_, x)
    assert(np.allclose(grad.data, grad_.data))

class TestCubCumsumOp(unittest.TestCase):
    def setUp(self):
        self.is_reversed = False

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_1d(self):
        check_forward([20])
        check_forward([6007])
        check_forward([6007], 0)
        check_forward([6007], -1)

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_1d_backward(self):
        check_backward([20])
        check_backward([6007])
        check_backward([6007], 0)
        check_backward([6007], -1)

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_2d(self):
        check_forward([5,5])
        check_forward([2000, 6007])
        check_forward([2000, 6007], 1)
        check_forward([2000, 6007], -1)

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_2d_backward(self):
        check_backward([5,5])
        check_backward([2000, 6007])
        check_backward([2000, 6007], 1)
        check_backward([2000, 6007], -1)

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_nd(self):
        check_forward([5,6,7,8], 0)
        check_forward([5,6,7,8], 1)
        check_forward([5,6,7,8], 2)
        check_forward([5,6,7,8], 3)
        check_forward([5,6,7,8], -1)
        check_forward([16,14,14,4096], 0)
        check_forward([16,14,14,4096], 1)
        check_forward([16,14,14,4096], 2)
        check_forward([16,14,14,4096], 3)
        check_forward([16,14,14,4096], -1)
        check_forward([16,14,14,4095], 3)

    @unittest.skipIf(cub_ops==None, "CUDA cub is unavailable in this build")
    @jt.flag_scope(use_cuda=1)
    def test_nd_backward(self):
        check_backward([5,6,7,8], 0)
        check_backward([5,6,7,8], 1)
        check_backward([5,6,7,8], 2)
        check_backward([5,6,7,8], 3)
        check_backward([5,6,7,8], -1)
        check_backward([16,14,14,4096], 0)
        check_backward([16,14,14,4096], 1)
        check_backward([16,14,14,4096], 2)
        check_backward([16,14,14,4096], 3)
        check_backward([16,14,14,4096], -1)
        check_backward([16,14,14,4095], 3)

if __name__ == "__main__":
    unittest.main()