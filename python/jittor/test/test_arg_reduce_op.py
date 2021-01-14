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
from .test_log import find_log_with_re
import copy
if jt.has_cuda:
    from jittor.compile_extern import cublas_ops, cudnn_ops, cub_ops
else:
    cublas_ops = cudnn_ops = cub_ops = None

def check_reduce(shape, op, dim, keepdims, is_cuda = False):
    with jt.log_capture_scope(
        log_silent=1,
        log_v=0, log_vprefix="op.cc=100"
    ) as raw_log:
        x = jt.random(shape)
        key, v = jt.arg_reduce(x, op, dim, keepdims)
        x_ = x.data
        key_ = key.data
        v_ = v.data
    if (is_cuda):
        logs = find_log_with_re(raw_log, "(Jit op key (not )?found: " + "cub_arg_reduce" + ".*)")
        assert len(logs)==1
    if op == 'max':
        key__ = np.argmax(x_, axis=dim)
        v__ = np.max(x_, axis=dim)
    else:
        key__ = np.argmin(x_, axis=dim)
        v__ = np.min(x_, axis=dim)
        
    if keepdims:
        key__ = np.expand_dims(key__, axis=dim)
        v__ = np.expand_dims(v__, axis=dim)
    assert np.allclose(key_, key__)
    assert np.allclose(v_, v__)

def check_backward(shape, op, dim, keepdims):
    x = jt.random(shape)
    v, key = jt.arg_reduce(x, op, dim, keepdims)
    loss = (key * key).sum()
    gs = jt.grad(loss, x) / 2
    assert np.allclose((gs * x).data, (gs * gs).data)

class TestArgReduceOp(unittest.TestCase):
    def test_backward(self):
        check_backward([5,5,5], 'min', 0, True)
        check_backward([5,5,5], 'min', 2, True)
        check_backward([5,5,5], 'min', 1, True)
        check_backward([5,], 'min', 0, True)
        check_backward([20,20,20,20], 'max', 0, True)
        check_backward([20,20,20,20], 'max', 2, True)
        check_backward([20,20,20,20], 'max', 1, True)
        check_backward([20,20,20,20], 'max', 3, True)
        check_backward([5,5,5], 'min', 0, False)
        check_backward([5,5,5], 'min', 2, False)
        check_backward([5,5,5], 'min', 1, False)
        check_backward([5,], 'min', 0, False)
        check_backward([20,20,20,20], 'max', 0, False)
        check_backward([20,20,20,20], 'max', 2, False)
        check_backward([20,20,20,20], 'max', 1, False)
        check_backward([20,20,20,20], 'max', 3, False)

    @unittest.skipIf(cub_ops==None, "Not use cub, Skip")
    @jt.flag_scope(use_cuda=1)
    def test_backward_cuda(self):
        check_backward([5,5,5], 'min', 0, True)
        check_backward([5,5,5], 'min', 2, True)
        check_backward([5,5,5], 'min', 1, True)
        check_backward([5,], 'min', 0, True)
        check_backward([20,20,20,20], 'max', 0, True)
        check_backward([20,20,20,20], 'max', 2, True)
        check_backward([20,20,20,20], 'max', 1, True)
        check_backward([20,20,20,20], 'max', 3, True)
        check_backward([5,5,5], 'min', 0, False)
        check_backward([5,5,5], 'min', 2, False)
        check_backward([5,5,5], 'min', 1, False)
        check_backward([5,], 'min', 0, False)
        check_backward([20,20,20,20], 'max', 0, False)
        check_backward([20,20,20,20], 'max', 2, False)
        check_backward([20,20,20,20], 'max', 1, False)
        check_backward([20,20,20,20], 'max', 3, False)

    def test(self):
        check_reduce([5,5,5], 'min', 0, True)
        check_reduce([5,5,5], 'min', 2, True)
        check_reduce([5,5,5], 'min', 1, True)
        check_reduce([5], 'min', 0, True)
        check_reduce([20,20,20,20], 'max', 0, True)
        check_reduce([20,20,20,20], 'max', 2, True)
        check_reduce([20,20,20,20], 'max', 1, True)
        check_reduce([20,20,20,20], 'max', 3, True)
        check_reduce([5,5,5], 'min', 0, False)
        check_reduce([5,5,5], 'min', 2, False)
        check_reduce([5,5,5], 'min', 1, False)
        check_reduce([5], 'min', 0, False)
        check_reduce([20,20,20,20], 'max', 0, False)
        check_reduce([20,20,20,20], 'max', 2, False)
        check_reduce([20,20,20,20], 'max', 1, False)
        check_reduce([20,20,20,20], 'max', 3, False)

    @unittest.skipIf(cub_ops==None, "Not use cub, Skip")
    @jt.flag_scope(use_cuda=1)
    def test_cuda(self):
        check_reduce([5,5,5], 'min', 0, True, True)
        check_reduce([5,5,5], 'min', 2, True, True)
        check_reduce([5,5,5], 'min', 1, True, True)
        check_reduce([5], 'min', 0, True)
        check_reduce([20,20,20,20], 'max', 0, True, True)
        check_reduce([20,20,20,20], 'max', 2, True, True)
        check_reduce([20,20,20,20], 'max', 1, True, True)
        check_reduce([20,20,20,20], 'max', 3, True, True)
        check_reduce([5,5], 'min', 0, False, True)
        check_reduce([5,5,5], 'min', 2, False, True)
        check_reduce([5,5,5], 'min', 1, False, True)
        check_reduce([5], 'min', 0, False)
        check_reduce([20,20,20,20], 'max', 0, False, True)
        check_reduce([20,20,20,20], 'max', 2, False, True)
        check_reduce([20,20,20,20], 'max', 1, False, True)
        check_reduce([20,20,20,20], 'max', 3, False, True)
if __name__ == "__main__":
    unittest.main()
