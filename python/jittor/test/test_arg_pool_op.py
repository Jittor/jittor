# ***************************************************************
# Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor.nn import Pool, pool, AvgPool2d, avg_pool2d
from jittor.nn import MaxPool2d as j_MaxPool2d
from jittor.nn import max_pool2d as j_max_pool2d
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

class OldPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None
        self.kernel_size = kernel_size
        self.op = op
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad and padding != 0

    def execute(self, x):
        N,C,H,W = x.shape
        if self.ceil_mode == False:
            h = (H+self.padding*2-self.kernel_size)//self.stride+1
            w = (W+self.padding*2-self.kernel_size)//self.stride+1
        else:
            h = (H+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1
            w = (W+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1

        # TODO: backward 
        xx = x.reindex([N,C,h,w,self.kernel_size,self.kernel_size], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.stride}-{self.padding}+i4", # Hid
            f"i3*{self.stride}-{self.padding}+i5", # Wid
        ])
        return xx.reduce(self.op, [4,5])


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
        shape = [2, 3, 300, 300]
        check(jt_model, torch_model, shape, False)
        shape = [2, 3, 157, 300]
        check(jt_model, torch_model, shape, False)
        for i in range(10):
            check(jt_model, torch_model, [1,1,300,300], True)

    @unittest.skipIf(not jt.compiler.has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda_tuple(self):
        jt_model = jt.nn.Sequential(Pool((2,3), (2,3), (1,1)), Pool((2,3), (2,3), (1,1)), Pool((2,3), (2,3), (1,1), ceil_mode=True), Pool((2,3), (2,3), (1,1)), Pool((2,3), (2,3), (1,1)), Pool(3, 1, 1))
        torch_model = Sequential(MaxPool2d((2,3), (2,3), (1,1)), MaxPool2d((2,3), (2,3), (1,1)), MaxPool2d((2,3), (2,3), (1,1), ceil_mode=True), MaxPool2d((2,3), (2,3), (1,1)), MaxPool2d((2,3), (2,3), (1,1)), MaxPool2d(3, 1, 1))
        shape = [2, 3, 300, 300]
        check(jt_model, torch_model, shape, False)
        shape = [2, 3, 157, 300]
        check(jt_model, torch_model, shape, False)
        for i in range(10):
            check(jt_model, torch_model, [1,1,300,300], True)

    @unittest.skipIf(True, "TODO: cannot pass this test, fix me")
    @unittest.skipIf(not jt.compiler.has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda_old_pool(self):
        from torch.nn import AvgPool2d
        jt_model = OldPool(3, 1, 1, op="mean")
        torch_model = AvgPool2d(3, 1, 1)
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
    
    def test_cpu_tuple(self):
        jt_model = jt.nn.Sequential(Pool((2,3), (2,3), (1,1)), Pool((2,3), (2,3), (1,1)), Pool((2,3), (2,3), (1,1), ceil_mode=True), Pool((2,3), (2,3), (1,1)), Pool((2,3), (2,3), (1,1)), Pool(3, 1, 1))
        torch_model = Sequential(MaxPool2d((2,3), (2,3), (1,1)), MaxPool2d((2,3), (2,3), (1,1)), MaxPool2d((2,3), (2,3), (1,1), ceil_mode=True), MaxPool2d((2,3), (2,3), (1,1)), MaxPool2d((2,3), (2,3), (1,1)), MaxPool2d(3, 1, 1))
        shape = [2, 3, 300, 300]
        check(jt_model, torch_model, shape, False)
        shape = [2, 3, 157, 300]
        check(jt_model, torch_model, shape, False)
        for i in range(10):
            check(jt_model, torch_model, [1,1,300,300], True)

    def test_index_pool(self):
        pool = jt.nn.Pool(2, return_indices=True)
        a = jt.randn([10,3,100,100])
        b, idx = pool(a)
        idx.sync()

    def test_index_pool2(self):
        pool = jt.nn.Pool(2, return_indices=True)
        a = jt.array([1,0,0,1,
                      0,0,0,0,
                      0,0,0,0,
                      1,0,0,1]).reshape((1,1,4,4))
        b, idx = pool(a)
        assert (idx.data.reshape((4,)) == [0,3,12,15]).all()

    def test_unpool(self):
        from jittor import nn
        pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        unpool = nn.MaxUnpool2d(2, stride=2)
        input = jt.array([[[[ 1.,  2,  3,  4,0],
                                [ 5,  6,  7,  8,0],
                                [ 9, 10, 11, 12,0],
                                [13, 14, 15, 16,0],
                                [0,  0,  0,  0, 0]]]])
        output, indices = pool(input)
        assert (indices == jt.array([[6,8],[16,18]])).all()
        out = unpool(output, indices, output_size=input.shape)
        assert (out == jt.array([[[[   0.,  0.,   0.,   0.,   0.],
                    [   0.,  6.,   0.,   8.,   0.],
                    [   0.,  0.,   0.,   0.,   0.],
                    [   0., 14.,   0.,  16.,   0.],
                    [   0.,  0.,   0.,   0.,   0.]]]])).all()

    def test_unpool_diff_kernel_stride(self):
        from jittor import nn
        pool = nn.MaxPool2d(3, stride=2, return_indices=True)
        unpool = nn.MaxUnpool2d(3, stride=2)
        input = jt.array([[[[ 1.,  2,  3,  4, 0],
                            [ 5,   6,  7,  8, 0],
                                [ 9, 10, 11, 12,0],
                                [13, 14, 16, 15,0],
                                [0,  0,  0,  0, 0]]]])
        output, indices = pool(input)
        out = unpool(output, indices, output_size=input.shape)
        assert (out == jt.array([[[
            [ 0.,  0.,  0.,  0.,  0.,],
            [ 0.,  0.,  0.,  0.,  0.,],
            [ 0.,  0., 11., 12.,  0.,],
            [ 0.,  0., 32.,  0.,  0.,],
            [ 0.,  0.,  0.,  0.,  0.,]]]])).all()

        

    @unittest.skipIf(not jt.compiler.has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda_avg_pool(self):
        self.test_cpu_avg_pool()

    def test_cpu_avg_pool(self):
        from torch.nn import AvgPool2d
        jt_model = Pool(2, 2, 0, op="mean", ceil_mode=True)
        torch_model = AvgPool2d(2, 2, 0, ceil_mode=True)
        shape = (2, 16, 33, 33)
        check(jt_model, torch_model, shape, False)

    def test_cpu_avg_pool2(self):
        from torch.nn import AvgPool2d
        jt_model = Pool(3, 1, 1, op="mean", ceil_mode=True)
        torch_model = AvgPool2d(3, 1, 1, ceil_mode=True)
        shape = (2, 16, 33, 33)
        check(jt_model, torch_model, shape, False)

    def test_AvgPool2d(self):
        from torch.nn import AvgPool2d as t_AvgPool2d
        jt_model = AvgPool2d(3, 1, 1, ceil_mode=True)
        torch_model = t_AvgPool2d(3, 1, 1, ceil_mode=True)
        shape = (2, 16, 33, 33)
        check(jt_model, torch_model, shape, False)

        jt_model = AvgPool2d(3, 1, 1, ceil_mode=True, count_include_pad=False)
        torch_model = t_AvgPool2d(3, 1, 1, ceil_mode=True, count_include_pad=False)
        shape = (2, 16, 100, 100)
        check(jt_model, torch_model, shape, False)
        print('finish')

    def test_avg_pool2d(self):
        from torch.nn.functional import avg_pool2d as t_avg_pool2d
        arr = np.random.random((2, 16, 33, 33))
        jt_model = avg_pool2d(jt.array(arr), 3, 1, 1, ceil_mode=True)
        torch_model = t_avg_pool2d(torch.Tensor(arr), 3, 1, 1, ceil_mode=True)
        assert np.allclose(jt_model.numpy(), torch_model.numpy())

        jt_model = avg_pool2d(jt.array(arr), 3, 1, 1, ceil_mode=True, count_include_pad=False)
        torch_model = t_avg_pool2d(torch.Tensor(arr), 3, 1, 1, ceil_mode=True, count_include_pad=False)
        assert np.allclose(jt_model.numpy(), torch_model.numpy())
        print('finish')

    def test_MaxPool2d(self):
        from torch.nn import MaxPool2d
        jt_model = j_MaxPool2d(3, 1, 1, ceil_mode=True)
        torch_model = MaxPool2d(3, 1, 1, ceil_mode=True)
        shape = (2, 16, 33, 33)
        check(jt_model, torch_model, shape, False)
        print('finish')

    def test_max_pool2d(self):
        from torch.nn.functional import max_pool2d
        arr = np.random.random((2, 16, 33, 33))
        jt_model = j_max_pool2d(jt.array(arr), 3, 1, 1, ceil_mode=True)
        torch_model = max_pool2d(torch.Tensor(arr), 3, 1, 1, ceil_mode=True)
        assert np.allclose(jt_model.numpy(), torch_model.numpy())

        jt_model = j_max_pool2d(jt.array(arr), 3, 1, 1)
        torch_model = max_pool2d(torch.Tensor(arr), 3, 1, 1)
        assert np.allclose(jt_model.numpy(), torch_model.numpy())

    def test_pool_3d(self):
        from torch.nn.functional import max_pool2d
        arr = np.random.random((2, 16, 20, 20, 20)).astype("float32")
        # arr = np.random.random((1, 1, 1, 5, 5)).astype("float32")
        jin = jt.array(arr)
        tin = torch.Tensor(arr)
        tin.requires_grad = True
        jt_model = jt.nn.Pool3d(3,1,1)(jin)
        torch_model = torch.nn.MaxPool3d(3,1,1)(tin)
        assert np.allclose(jt_model.numpy(), torch_model.detach().numpy())


        nout = np.random.random(tuple(jt_model.shape)).astype("float32")
        jout = jt_model * nout
        tout = torch_model * torch.Tensor(nout)
        dj = jt.grad(jout, jin)
        
        tout.sum().backward()
        dt = tin.grad
        assert np.allclose(dj.numpy(), dt.numpy())

    @unittest.skipIf(not jt.compiler.has_cuda, "No cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda_pool_3d(self):
        self.test_pool_3d()


if __name__ == "__main__":
    unittest.main()