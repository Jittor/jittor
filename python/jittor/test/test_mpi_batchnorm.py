# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os, sys
import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from jittor.test.test_mpi import run_mpi_test

mpi = jt.compile_extern.mpi
if mpi:
    n = mpi.world_size()

class FakeMpiBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=None, is_train=True):
        assert affine == None

        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.weight = init.constant((num_features,), "float32", 1.0)
        self.bias = init.constant((num_features,), "float32", 0.0)
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()

    def execute(self, x, global_x):
        if self.is_train:
            xmean = jt.mean(global_x, dims=[0,2,3], keepdims=1)
            x2mean = jt.mean(global_x*global_x, dims=[0,2,3], keepdims=1)

            xvar = x2mean-xmean*xmean
            norm_x = (x-xmean)/jt.sqrt(xvar+self.eps)
            self.running_mean.update(self.running_mean + 
                (xmean.sum([0,2,3])-self.running_mean)*self.momentum)
            self.running_var.update(self.running_var + 
                (xvar.sum([0,2,3])-self.running_var)*self.momentum)
        else:
            running_mean = self.running_mean.broadcast(x, [0,2,3])
            running_var = self.running_var.broadcast(x, [0,2,3])
            norm_x = (x-running_mean)/jt.sqrt(running_var+self.eps)
        w = self.weight.broadcast(x, [0,2,3])
        b = self.bias.broadcast(x, [0,2,3])
        return norm_x * w + b

@unittest.skipIf(not jt.in_mpi, "no inside mpirun")
class TestMpiBatchnorm(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        jt.seed(3)

    def test_batchnorm(self):
        mpi = jt.compile_extern.mpi
        data = np.random.rand(30,3,10,10).astype("float32")
        x1 = jt.array(data)
        stride = 30//n
        x2 = jt.array(data[mpi.world_rank()*stride:(mpi.world_rank()+1)*stride,...])
        
        bn1 = nn.BatchNorm(3, sync=False)
        bn2 = nn.BatchNorm(3, sync=True)
        bn3 = FakeMpiBatchNorm(3)
        y1 = bn1(x1).data
        y2 = bn2(x2).data
        y3 = bn3(x2,x1).data

        assert np.allclose(y2, y3, atol=1e-4), (y2, y3)
        assert np.allclose(bn1.running_mean.data, bn2.running_mean.data), \
            (bn1.running_mean.data, bn2.running_mean.data)
        assert np.allclose(bn1.running_var.data, bn2.running_var.data)
        
    def test_batchnorm_backward(self):
        mpi = jt.compile_extern.mpi
        data = np.random.rand(30,3,10,10).astype("float32")
        global_x = jt.array(data)
        stride = 30//n
        x = jt.array(data[mpi.world_rank()*stride:(mpi.world_rank()+1)*stride,...])
        
        bn1 = nn.BatchNorm(3, sync=True)
        bn2 = FakeMpiBatchNorm(3)
        y1 = bn1(x)
        y2 = bn2(x,global_x)
        gs1 = jt.grad(y1,bn1.parameters())
        gs2 = jt.grad(y2,bn2.parameters())

        assert np.allclose(y1.data, y2.data, atol=1e-5),(mpi.world_rank(),y1.data, y2.data, y1.data-y2.data)
        assert len(gs1) == len(gs2)
        for i in range(len(gs1)):
            assert np.allclose(gs1[i].data, gs2[i].data, rtol=1e-2),(mpi.world_rank(),gs1[i].data, gs2[i].data,gs1[i].data-gs2[i].data)

    @unittest.skipIf(not jt.has_cuda, "no cuda")
    @jt.flag_scope(use_cuda=1)
    def test_batchnorm_cuda(self):
        self.test_batchnorm()
        self.test_batchnorm_backward()


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiBatchnormEntry(unittest.TestCase):
    def test(self):
        run_mpi_test(2, "test_mpi_batchnorm")

if __name__ == "__main__":
    unittest.main()
