# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
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
if jt.has_cuda:
    from jittor.compile_extern import cublas_ops, cudnn_ops, cub_ops
else:
    cublas_ops = cudnn_ops = cub_ops = None

def check_argsort(shape, dim, descending = False):
    x = jt.random(shape)
    y, y_key = jt.argsort(x, dim=dim, descending=descending)
    v = []
    for i in range(len(shape)):
        if (i == dim):
            v.append(y)
        else:
            v.append(jt.index(shape, dim=i))
    yk = jt.reindex(x, v)
    yk_ = yk.data
    y_key_ = y_key.data
    x__ = x.data
    if descending:
        x__ = -x__
    yk__ = np.sort(x__, axis=dim)
    if descending:
        yk__ = -yk__
    assert np.allclose(y_key_, yk__)
    assert np.allclose(yk_, yk__)

def check_cub_argsort(shape, dim, descending = False):
    with jt.log_capture_scope(
        log_silent=1,
        log_v=0, log_vprefix="op.cc=100"
    ) as raw_log:
        x = jt.random(shape)
        y, y_key = jt.argsort(x, dim=dim, descending=descending)
        v = []
        for i in range(len(shape)):
            if (i == dim):
                v.append(y)
            else:
                v.append(jt.index(shape, dim=i))
        yk = jt.reindex(x, v)
        yk_ = yk.data
        y_key_ = y_key.data
    logs = find_log_with_re(raw_log, "(Jit op key (not )?found: " + "cub_argsort" + ".*)")
    assert len(logs)==1
    x__ = x.data
    if descending:
        x__ = -x__
    yk__ = np.sort(x__, axis=dim)
    if descending:
        yk__ = -yk__
    assert np.allclose(y_key_, yk__)
    assert np.allclose(yk_, yk__)

def check_backward(shape, dim, descending = False):
    x = jt.random(shape)
    y, y_key = jt.argsort(x, dim=dim, descending=descending)
    loss = (y_key * y_key).sum()
    gs = jt.grad(loss, x)
    assert np.allclose(x.data*2, gs.data)

class TestArgsortOp(unittest.TestCase):
    def test(self):
        check_argsort([5,5], 0, False)
        check_argsort([5,5], 0, True)
        check_argsort([5,5], 1, False)
        check_argsort([5,5], 1, True)
        check_argsort([12, 34, 56, 78], 1, True)
        check_argsort([12, 34, 56, 78], 3, True)
        check_argsort([12, 34, 56, 78], 2, False)
        check_argsort([12, 34, 56, 78], 0, False)

    def test_backward(self):
        check_backward([5,5], 0, False)
        check_backward([5,5], 0, True)
        check_backward([5,5], 1, False)
        check_backward([5,5], 1, True)
        check_backward([12, 34, 56, 78], 1, True)
        check_backward([12, 34, 56, 78], 3, True)
        check_backward([12, 34, 56, 78], 2, False)
        check_backward([12, 34, 56, 78], 0, False)

    def test_doc(self):
        assert "Argsort Operator" in jt.argsort.__doc__

    @unittest.skipIf(cub_ops==None, "Not use cub, Skip")
    @jt.flag_scope(use_cuda=1)
    def test_cub(self):
        check_cub_argsort([5,5], 0, False)
        check_cub_argsort([5,5], 0, True)
        check_cub_argsort([5,5], 1, False)
        check_cub_argsort([5,5], 1, True)
        check_cub_argsort([12, 34, 56, 78], 1, True)
        check_cub_argsort([12, 34, 56, 78], 3, True)
        check_cub_argsort([12, 34, 56, 78], 2, False)
        check_cub_argsort([12, 34, 56, 78], 0, False)

    @unittest.skipIf(cub_ops==None, "Not use cub, Skip")
    @jt.flag_scope(use_cuda=1)
    def test_cub_backward(self):
        check_backward([5,5], 0, False)
        check_backward([5,5], 0, True)
        check_backward([5,5], 1, False)
        check_backward([5,5], 1, True)
        check_backward([12, 34, 56, 78], 1, True)
        check_backward([12, 34, 56, 78], 3, True)
        check_backward([12, 34, 56, 78], 2, False)
        check_backward([12, 34, 56, 78], 0, False)

if __name__ == "__main__":
    unittest.main()