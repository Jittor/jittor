# ***************************************************************
# Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor.nn import Pool, pool
import numpy as np
from .test_core import expect_error
from .test_grad import ngrad
from itertools import permutations
from jittor import compile_extern, Module
from .test_log import find_log_with_re
import random
import pickle as pk

skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    from torch.nn import MaxPool2d, Sequential
except:
    skip_this_test = True

def check(jt_model, torch_model, shape, near_data):
    if (near_data):
        assert shape[0] * shape[1] * shape[2] * shape[3] % 8 == 0
        data = list(range(8)) * int((shape[0] * shape[1] * shape[2] * shape[3]) / 8)
        random.shuffle(data)
        x = jt.array(data).float32().reshape(shape)
    else:
        x = jt.random(shape)
    y = jt_model(x)
    g = jt.grad(y.sum(), x)

    x_ = torch.Tensor(x.data)
    x_.requires_grad = True
    y_ = torch_model(x_)
    y_.sum().backward()
    y__ = y_.detach().numpy()
    g__ = x_.grad.detach().numpy()
    assert np.allclose(y.data, y__)
    assert np.allclose(g.data, g__)

@unittest.skipIf(skip_this_test, "No Torch found")
class TestArgPoolOp(unittest.TestCase):
    @unittest.skipIf(not jt.compiler.has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda(self):
        jt_model = jt.nn.Sequential(Pool(2, 2, 0), Pool(2, 2, 0), Pool(2, 2, 0, ceil_mode=True), Pool(2, 2, 0), Pool(2, 2, 0), Pool(3, 1, 1))
        torch_model = Sequential(MaxPool2d(2, 2, 0), MaxPool2d(2, 2, 0), MaxPool2d(2, 2, 0, ceil_mode=True), MaxPool2d(2, 2, 0), MaxPool2d(2, 2, 0), MaxPool2d(3, 1, 1))
        shape = [64, 64, 300, 300]
        check(jt_model, torch_model, shape, False)
        shape = [32, 128, 157, 300]
        check(jt_model, torch_model, shape, False)
        for i in range(10):
            check(jt_model, torch_model, [1,1,300,300], True)
            
    def test_cpu_(self):
        # x = jt.random([32, 128, 157, 300])
        x = jt.random([4, 128, 157, 300])
        x = jt.nn.pool(x, 2, "maximum", 0, 2)

    def test_cpu(self):
        jt_model = jt.nn.Sequential(Pool(2, 2, 0), Pool(2, 2, 0), Pool(2, 2, 0, ceil_mode=True), Pool(2, 2, 0), Pool(2, 2, 0), Pool(3, 1, 1))
        torch_model = Sequential(MaxPool2d(2, 2, 0), MaxPool2d(2, 2, 0), MaxPool2d(2, 2, 0, ceil_mode=True), MaxPool2d(2, 2, 0), MaxPool2d(2, 2, 0), MaxPool2d(3, 1, 1))
        # shape = [64, 64, 300, 300]
        shape = [4, 64, 300, 300]
        check(jt_model, torch_model, shape, False)
        # shape = [32, 128, 157, 300]
        shape = [4, 128, 157, 300]
        check(jt_model, torch_model, shape, False)
        for i in range(10):
            check(jt_model, torch_model, [1,1,300,300], True)

    @unittest.skipIf(not jt.compiler.has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda_avg_pool(self):
        self.test_cpu_avg_pool()

    def test_cpu_avg_pool(self):
        from torch.nn import AvgPool2d
        jt_model = Pool(2, 2, 0, op="mean", ceil_mode=True)
        torch_model = AvgPool2d(2, 2, 0, ceil_mode=True)
        # shape = [64, 64, 300, 300]
        shape = (2, 16, 33, 33)
        check(jt_model, torch_model, shape, False)

if __name__ == "__main__":
    unittest.main()