
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
import time
import random
# import torch

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
    # def test_add(self):
    #     jt.flags.use_mlu=True
    #     jt.profiler.start(0, 1)
    #     a = np.random.randn(10, 1).astype(np.float32)
    #     aj = jt.array(a)
    #     b = np.random.randn(10, 1).astype(np.float32)
    #     bj = jt.array(b)
    #     print("after array")
    #     c = a+b
    #     cj = aj+bj
    #     # jt.sync_all(True)
    #     print("cj", cj.numpy())
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
    #     jt.flags.use_mlu=True
    #     jt.profiler.start(1, 0)

    #     # a = np.random.randn(1,16,224,224).astype(np.float32)
    #     a = np.ones((1,16,8,8)).astype(np.float32)
    #     aj = jt.array(a)
    #     bj=aj.sum(dims=[1])
    #     print(bj.numpy())
    #     return

    #     mj = jnn.BatchNorm(10, is_train=True)
    #     b = npBatchnorm(a)
    #     bj = jtBatchnorm(aj)
    #     nbj=bj.numpy()
    #     jt.profiler.stop()
    #     jt.profiler.report()
    #     # print("bj",nbj.shape,nbj)
    #     # print("b",b.shape,b)
    #     bj.numpy()
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

    # def test_batchnorm_eval(self):
    #     jt.flags.use_mlu=True
    #     jt.seed(0)
    #     np.random.seed(0)
    #     random.seed(0)
    #     na = np.random.randn(1,64,224,224).astype(np.float32)
    #     model = jt.nn.BatchNorm(64)
    #     model.eval()
        
    #     a = jt.array(na)
    #     b1 = model(a)
    #     jt.sync_all(True)
    #     a = jt.array(na)
    #     b2 = model(a)
        
    #     mb1=b1.numpy().mean()
    #     mb2=b2.numpy().mean()
    #     print("b1",mb1)
    #     print("b2",mb2)
    #     print("error:",abs(mb1-0.00018934582))
    #     assert mb1==mb2
    #     assert abs(mb1-0.00018934582)<1e-6

    # def test_linear(self):
    #     jt.flags.use_mlu=True

    #     jt.seed(0)
    #     np.random.seed(0)
    #     random.seed(0)
    #     sz=512
    #     sz2=1024
    #     na = np.random.randn(1, sz).astype(np.float32)
    #     model = jt.nn.Linear(sz, sz2)
    #     # nw = np.ones([sz2, sz]).astype(np.float32)
    #     # nw = np.random.randn(sz2, sz).astype(np.float32)
    #     # model.weight = jt.array(nw)
    #     # model.bias = jt.zeros((sz2,))
    #     model.eval()
        
    #     a = jt.array(na)
    #     b1 = model(a)
    #     jt.sync_all(True)
    #     jt.profiler.start(0, 0)
    #     a = jt.array(na)
    #     b2 = model(a)
    #     b2.numpy()
    #     jt.profiler.stop()
    #     jt.profiler.report()
        
    #     mb1=b1.numpy().mean()
    #     mb2=b2.numpy().mean()
    #     print("error:",abs(mb1-(-0.03882862)))
    #     assert mb1==mb2
    #     assert abs(mb1-(-0.03882862))<1e-6

    # def test_pool(self):
    #     jt.flags.use_mlu = True
    #     a= jt.array(np.random.randn(2,3,4,4).astype(np.float32))
    #     jt.profiler.start(0, 0)
    #     model = jt.nn.Pool(2, 2, 0, op="mean")
    #     b=model(a)
    #     print("b",b.numpy())
    #     print("a",a.numpy())
    #     jt.profiler.stop()
    #     jt.profiler.report()

    # def test_transpose(self):
    #     jt.flags.use_mlu = True
    #     a= jt.array(np.random.randn(2,3,3).astype(np.float32))
    #     jt.profiler.start(0, 0)
    #     b=a.transpose([1,0,2])
    #     print("b",b.shape,b.numpy())
    #     print("a",a.shape,a.numpy())
    #     jt.profiler.stop()
    #     jt.profiler.report()

    # def test_cast(self):
    #     jt.flags.use_mlu=True
    #     jt.seed(0)
    #     np.random.seed(0)
    #     random.seed(0)
    #     a=np.arange(64*224*224).astype(np.float32)
    #     b=jt.array(a)
    #     c=b.int8()
    #     c.numpy()
    #     jt.profiler.start(0, 0)
    #     b=jt.array(a)
    #     c=b.int8()
    #     print(c.numpy())
    #     jt.profiler.stop()
    #     jt.profiler.report()

    def test_resnet(self):
        # return
        jt.flags.use_mlu = True
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        seed = 0
        jt.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        a=np.random.randn(512,3,224,224).astype(np.float32)
        # a=np.random.randn(1,64,224,224).astype(np.float32)
        # a=np.ones((1,3,224,224)).astype(np.float32)
        # a=np.load("tmp_incpu.npy")
        # a=np.load("tmp_inmlu.npy")
        # a=a[:,:,:64,:64]
        # a=a[:,-2,:,:]
        # a=np.repeat(a,64,axis=0).reshape([1,64,112,112])
        # print(a.shape)
        # print(a)

        warmup=10
        rerun=10
        var = jt.array(a)
        print("var",var.shape)
        model = resnet.resnet18()
        model.load('resnet18-int8.pkl')
        model.train()
        for i in model.modules():
            if isinstance(i, jt.nn.BatchNorm):
                i.eps = 1e-4
        jt.sync_all(True)
        for i in range(warmup):
            # out = model(var).numpy()
            model(var).sync()
        jt.sync_all(True)
        jt.profiler.start(0, 0)
        time_start=time.time()
        for i in range(rerun):
            # out = model(var).numpy()
            model(var).sync()
        jt.sync_all(True)
        time_end=time.time()  
        print('avg time cost',(time_end-time_start)/rerun,'s')
        jt.profiler.stop()
        jt.profiler.report()
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
