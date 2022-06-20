
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
import numpy as np
import jittor as jt
from jittor import init, Module
import jittor.nn as jnn
from jittor.models import resnet
import os
import time
import random
# import torch

import ctypes

# ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libOpenCL.so", os.RTLD_NOW | os.RTLD_GLOBAL)

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

def optional_fake_half_cpu_inputs(tensor):
    if isinstance(tensor, tuple):
        tensor = tuple(x.type(torch.HalfTensor).type(torch.FloatTensor) for x in tensor)
    else:
        tensor = tensor.type(torch.HalfTensor).type(torch.FloatTensor)
        tensor[tensor == float("inf")] = 65504
    return tensor

def assertTensorsEqual(a, b, prec=None, message='', allow_inf=False, use_MSE=False, use_RAE=False, use_RMA=False):
    '''unittest.TestCase'''
    #if a.dtype == torch.bool:
    #    a = a.float()
    #if b.dtype == torch.bool:
    #    b = b.float()
    epsilon = 1.0 / 16384
    allow_inf = False
    #self.assertEqual(a.size(), b.size(), message)
    if a.numel() > 0:
        a = optional_fake_half_cpu_inputs(a)
        b = optional_fake_half_cpu_inputs(b)
        # check that NaNs are in the same locations
        nan_mask = a != a
        #self.assertTrue(torch.equal(nan_mask, b != b), message)
        diff = a - b
        diff[nan_mask] = 0
        # inf check if allow_inf=True
        if allow_inf:
            inf_mask = (a == float("inf")) | (a == float("-inf"))
            self.assertTrue(torch.equal(inf_mask,(b == float("inf")) | (b == float("-inf"))),message)
            diff[inf_mask] = 0
        # TODO: implement abs on CharTensor
        if diff.is_signed() and 'CharTensor' not in diff.type():
            diff = diff.abs()
        if use_MSE:
            diff = diff.abs().pow(2).sum()
            a_pow_sum = a.pow(2).sum()
            if diff <= (2 * epsilon) * (2 * epsilon):
                diff = 0.0
            if a_pow_sum <= epsilon:
                a_pow_sum += epsilon
            diff = torch.div(diff, a_pow_sum)
            print("diff:",diff.sqrt())
            #self.assertLessEqual(diff.sqrt(), prec, message)
        else:
            max_err = diff.max()
            self.assertLessEqual(max_err, prec, message)

class TestMLU(unittest.TestCase):
    def test_add(self):
        return
        jt.flags.use_device=True
        warmup=100
        rerun=1000
        a = np.random.randn(1, 1, 1, 16).astype(np.float32)
        b = np.random.randn(1, 1, 1, 16).astype(np.float32)
        aj = jt.array(a)
        bj = jt.array(b)
        # print("a",a)
        # print("b",b)
        c = a+b
        for i in range(warmup):
            cj = aj+bj
            cj.sync()
        jt.sync_all(True)
        print("start")
        jt.profiler.start(0, 0)
        time_start=time.time()
        for i in range(rerun):
            cj = aj+bj
            cj.sync()
        jt.sync_all(True)
        time_end=time.time()
        print("finish")
        print('tot time cost',time_end-time_start,'s')
        print("a",a)
        print("b",b)
        print("c",c)
        print("cj",cj.numpy())
        jt.profiler.stop()
        jt.profiler.report()

    # def test_sum(self):
    #     jt.flags.use_opencl=True
    #     jt.profiler.start(0, 0)
    #     a = np.random.randn(2, 4, 8).astype(np.float32)
    #     # a = np.arange(1, 10, 1).astype(np.float32)
    #     aj = jt.array(a)
    #     print("a",a)
    #     b = np.sum(a, axis=(1))
    #     bj = aj.sum([1,])
    #     # jt.sync_all(True)
    #     print("b", b.shape, b)
    #     print("bj", bj.shape, bj.numpy())
    #     # assert np.allclose(cj.numpy(), c,rtol=1e-5,atol=1e-5)
    #     jt.profiler.stop()
    #     jt.profiler.report()

    def test_conv(self):
        return 
        jt.flags.use_opencl=True
        seed = 0
        jt.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        a = np.ones((1, 1, 3, 3)).astype(np.float32)
        aj = jt.array(a)
        # jt.profiler.start(0, 0)
        # cj = aj.mean()
        # print("cj", cj.shape, cj.numpy())
        # jt.profiler.stop()
        # jt.profiler.report()
        # return

        conv = jt.nn.Conv(1, 1, kernel_size=3, padding=1, bias=False)
        # jt.init.constant_(conv.weight, 1.0)
        print("a",a)
        # print("weight",conv.weight.numpy())
        jt.profiler.start(0, 0)
        bj = conv(aj)
        print("bj", bj.shape, bj.numpy())
        cj = jt.array(bj.numpy()).mean()
        print("cj", cj.shape, cj.numpy())
        jt.profiler.stop()
        jt.profiler.report()

    def test_bn(self):
        return
        jt.flags.use_opencl=True
        seed = 0
        jt.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        a = np.random.rand(1, 64, 112, 112).astype(np.float32)
        aj = jt.array(a)
        # jt.profiler.start(0, 0)
        # cj = aj.mean()
        # print("cj", cj.shape, cj.numpy())
        # jt.profiler.stop()
        # jt.profiler.report()
        # return

        bn = jt.nn.BatchNorm(64)
        bn.eval()
        print("a",a)
        jt.profiler.start(0, 0)
        bj = bn(aj)
        print("bj", bj.shape, bj.sum().numpy())
        jt.profiler.stop()
        jt.profiler.report()

    def test_resnet(self):
        # return
        jt.flags.use_device = False
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        seed = 0
        jt.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        a=np.random.randn(1,3,224,224).astype(np.float32)

        var = jt.array(a)
        print("var",var.shape)
        # bn = jt.nn.BatchNorm(64)
        # bn.eval()
        model = resnet.resnet18()
        model.eval()
        # model.load('resnet18.npy')
        # jt.sync_all(True)
        # model(var).sync()
        # SGD = jt.nn.SGD(model.parameters(), 0.1, 0.9, 0.0001)
        warmup=10
        rerun=10
        print("warmup")
        # jt.sync_all(True)
        for i in range(warmup):
            # print(i)
            loss = model(var)
            loss.sync()
            # print("loss",loss.sum().numpy())
        jt.sync_all(True)
        
        print("run")
        jt.profiler.start(0, 0)
        # time_start=time.time()
        for i in range(rerun):
            # print(i)
            loss = model(var)
            loss.sync()
            # print("loss",loss.sum().numpy())
        jt.sync_all(True)
        # time_end=time.time()
        # print("finish")
        # spd=(time_end-time_start)/rerun
        # print('tot time cost',time_end-time_start,'s')
        # print('avg time cost',spd,'s')
        # print('avg fps',1.0/spd)
        jt.profiler.stop()
        jt.profiler.report()
        
        return

        # jt.profiler.start(0, 0)
        # time_start=time.time()
        # for i in range(rerun):
        #     # out = model(var).numpy()
        #     loss = model(var).mean()
        #     SGD.step(loss)
        #     jt.sync_all()
        # jt.sync_all(True)
        # time_end=time.time()  
        # print('avg time cost',(time_end-time_start)/rerun,'s')
        # jt.profiler.stop()
        # jt.profiler.report()
        # print("out",out)
        # # np.save("tmp.npy",out)
        # sout=np.load("tmp.npy")
        # print("std",sout)
        # print("error",np.abs(sout-out))
        # print("max error",np.abs(sout-out).max())
        # print("mse",((sout-out)**2).mean())
        # assertTensorsEqual(torch.tensor(sout),torch.tensor(out),0.02,use_MSE = True)
        

        
if __name__ == "__main__":
    unittest.main()
