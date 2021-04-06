
# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import numpy as np
import jittor as jt
from jittor import init, Module
import jittor.nn as jnn
from jittor.models import resnet
import time

def npBatchnorm(x):
    eps=1e-5
    dims = [0]+list(range(2,x.ndim))
    xmean = x.transpose(1,0,2,3).reshape(x.shape[1],-1)
    xmean = np.mean(xmean, axis=1)
    x2mean = (x*x).transpose(1,0,2,3).reshape(x.shape[1],-1)
    x2mean = np.mean(x2mean, axis=1)

    xvar = np.maximum((x2mean-xmean*xmean), 0.0)
    w = 1.0 / np.sqrt(xvar+eps)
    b = 0.0 - xmean * w
    for i in dims:
        w = np.expand_dims(w, i)
        b = np.expand_dims(b, i)
    norm_x = x * w + b

    return norm_x

def jtBatchnorm(x):
    eps=1e-5
    dims = [0]+list(range(2,x.ndim))
    # return jt.mean(x, dims=dims)
    xmean = jt.mean(x, dims=dims)
    # xmean = x[0,:,0,0]
    # print("xmean",xmean.numpy())
    x2mean = jt.mean(x*x, dims=dims)
    # print("x2mean",x2mean.numpy())

    xvar = (x2mean-xmean*xmean).maximum(0.0)
    # print("xvar",xvar.numpy())
    w = 1.0 / jt.sqrt(xvar+eps)
    # print("w",w.numpy())
    b = 0.0 - xmean * w
    # print("b",b.numpy())
    norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)

    return norm_x

class TestMLU(unittest.TestCase):
    # def test_add(self):
    #     jt.profiler.start(0, 1)
    #     a = np.random.randn(10, 1).astype(np.float32)
    #     aj = jt.array(a)
    #     b = np.random.randn(10, 1).astype(np.float32)
    #     bj = jt.array(b)
    #     c = a+b
    #     cj = aj+bj
    #     assert np.allclose(cj.numpy(), c,rtol=1e-5,atol=1e-5)
    #     jt.profiler.stop()
    #     jt.profiler.report()

    # def test_sub(self):
    #     jt.profiler.start(0, 1)
    #     a = np.random.randn(10, 1).astype(np.float32)
    #     aj = jt.array(a)
    #     b = np.random.randn(10, 1).astype(np.float32)
    #     bj = jt.array(b)
    #     c = a-b
    #     cj = aj-bj
    #     assert np.allclose(cj.numpy(), c,rtol=1e-5,atol=1e-5)
    #     jt.profiler.stop()
    #     jt.profiler.report()
        
        
    # def test_batchnorm(self):
    #     jt.profiler.start(0, 1)
    #     a = np.random.randn(16,10,224,224).astype(np.float32)
    #     # a = np.ones([1,10,3,3]).astype(np.float32)
    #     aj = jt.array(a)
    #     mj = jnn.BatchNorm(10, is_train=True)
    #     b = npBatchnorm(a)
    #     # print("b",b)
    #     bj = jtBatchnorm(aj)
    #     # print("bj",bj.numpy())
    #     jt.profiler.stop()
    #     jt.profiler.report()
    #     print("batchnorm error",np.max(bj.numpy()-b))
    #     assert np.allclose(bj.numpy(), b,rtol=1e-1,atol=1e-1)

    # def test_conv(self):
    #     jt.profiler.start(0, 1)
    #     # for i in range(100):
    #     # a = np.random.randn(160000,3,3,3).astype(np.float32)
    #     a = np.ones([160000,3,3,3]).astype(np.float32)
    #     a = jt.array(a)
    #     model = jnn.Conv(3,3,3, padding=1, bias=False)
    #     model.weight.assign(jt.ones(model.weight.shape))
    #     b = model(a)
    #     print("conv result", b.shape,b.numpy())
    #     jt.profiler.stop()
    #     jt.profiler.report()

    def test_resnet(self):
        na = np.random.randn(1,3,224,224).astype(np.float32)
        model = resnet.Resnet18()
        model.eval()
        
        a = jt.array(na)
        b = model(a)
        print("resnet result", b.shape,b.numpy())
        jt.sync_all(True)

        print("time start")
        # jt.profiler.start(0, 1)
        time_start=time.time()
        # a = np.random.randn(1,3,224,224).astype(np.float32)
        a = jt.array(na)
        b = model(a)
        print("resnet result", b.shape,b.numpy())
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        # jt.profiler.stop()
        # jt.profiler.report()

        
if __name__ == "__main__":
    unittest.main()