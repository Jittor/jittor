# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn, Module
from jittor.models import densenet
import numpy as np
import sys, os
import random
import math
import unittest
from jittor.test.test_reorder_tuner import simple_parser
from jittor.test.test_log import find_log_with_re
from jittor.dataset.mnist import MNIST
import jittor.transform as trans
import time

jt.flags.use_cuda = 1
skip_this_test = False

class InvFunc(jt.InvertibleFunction):
    def __init__(self, dim):
        self.dim = dim
        self.line = nn.Linear(dim, dim, bias=False)
    def execute(self, x1, x2):
        res1=x1
        res2=x2+self.line(x1)
        return res1, res2
    def invert_func(self, y1, y2):
        return y1, y2-self.line(y1)

class InvFunc2(jt.InvertibleFunction):
    def __init__(self, dim):
        self.dim = dim
        self.line = nn.Linear(dim, dim, bias=False)
    def execute(self, x1, x2):
        return self.line(x1) ,x2
    def invert_func(self, y1, y2):
        return self.line(y1), y2

class ForwardFunc(jt.Module):
    def __init__(self, dim):
        self.dim = dim
        self.line = nn.Linear(dim, dim, bias=False)
    def execute(self, x1, x2):
        return x1, x2+self.line(x1)

class ForwardFunc2(jt.Module):
    def __init__(self, dim):
        self.dim = dim
        self.line = nn.Linear(dim, dim, bias=False)

    def execute(self, x):
        return x*x

class BackwardFunc(jt.Module):
    def __init__(self, dim):
        self.dim = dim
        self.line = nn.Linear(dim, dim, bias=False)
    def execute(self, x1, x2):
        return x1, x2-self.line(x1)

class MultiInvFunc(jt.Module):
    def __init__(self, dim):
        self.dim=dim
        self.models = nn.Sequential()
        for i in range(200):
            self.models.append(InvFunc(dim))
    
    def execute(self, x1, x2):
        for i in range(len(self.models.layers)):
            x2,x1=self.models.layers[i](x1,x2)
        return x1,x2

class MultiForwardFunc(jt.Module):
    def __init__(self, dim):
        self.dim=dim
        self.models = nn.Sequential()
        for i in range(200):
            self.models.append(ForwardFunc(dim))
    
    def execute(self, x1, x2):
        for i in range(len(self.models.layers)):
            x2,x1=self.models.layers[i](x1,x2)
        return x1,x2

class MultiInvFunc2(jt.Module):
    def __init__(self, dim):
        self.models = nn.Sequential()
        for i in range(4):
            self.models.append(InvFunc2(dim))
    
    def execute(self, x1, x2):
        for i in range(len(self.models.layers)):
            x1, x2=self.models.layers[i](x1, x2)
        return x1, x2

class MultiForwardFunc2(jt.Module):
    def __init__(self, dim):
        self.models = nn.Sequential()
        for i in range(200):
            self.models.append(ForwardFunc2(dim))
    
    def execute(self, x):
        for i in range(len(self.models.layers)):
            x=self.models.layers[i](x)
        return x

@unittest.skipIf(skip_this_test, "skip_this_test")
class TestInvertibleOp(unittest.TestCase):
    # setup random seed
    def setup_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        jt.seed(seed)

    def test_intertible_op(self):
        # return
        self.setup_seed(2)
        x1 = jt.ones([1,3])
        x2 = jt.ones([1,3])
        func = InvFunc(3)
        ffunc = ForwardFunc(3)
        ffunc.line.weight.assign(func.line.weight.numpy())
        bfunc = BackwardFunc(3)
        bfunc.line.weight.assign(func.line.weight.numpy())

        y1,y2 = func(x1, x2)
        loss=y1+y2
        dx1, dx2 = jt.grad(loss, [x1, x2])
        x1 = jt.ones([1,3])
        x2 = jt.ones([1,3])
        fy1,fy2 = ffunc(x1, x2)
        floss=fy1+fy2
        fdx1, fdx2 = jt.grad(floss, [x1, x2])
        assert np.allclose(dx1.numpy(), fdx1.numpy())
        assert np.allclose(dx2.numpy(), fdx2.numpy())

        sy1 = y1.numpy()
        sy2 = y2.numpy()
        y1 = jt.array(sy1)
        y2 = jt.array(sy2)
        xx1, xx2 = func.invert(y1,y2)
        loss=xx1+xx2
        dy1, dy2 = jt.grad(loss, [y1, y2])
        y1 = jt.array(sy1)
        y2 = jt.array(sy2)
        by1,by2 = bfunc(y1, y2)
        bloss=by1+by2
        bdy1, bdy2 = jt.grad(bloss, [y1, y2])
        assert np.allclose(x1.numpy(), xx1.numpy())
        assert np.allclose(x2.numpy(), xx2.numpy())
        assert np.allclose(dy1.numpy(), bdy1.numpy())
        assert np.allclose(dy2.numpy(), bdy2.numpy())

    def test_memory(self):
        return
        start = time.time()
        dim=4
        if True:
            func = MultiInvFunc(dim)
        else:
            func = MultiForwardFunc(dim)
        print("func.parameters()",len(func.parameters()))
        SGD = nn.SGD(func.parameters(), 1e-3)
        for i in range(10):
            iter_start = time.time()
            print(i)
            x1 = jt.ones([4096,128,dim])
            x2 = jt.ones([4096,128,dim])
            y1,y2 = func(x1, x2)
            loss=y1+y2
            SGD.step(loss)
            # print("func.parameters()",len(func.parameters()))
            # dx1, dx2 = jt.grad(loss, [x1, x2])
            # dw = jt.grad(loss, func.parameters())
            # del x1
            # del x2
            # del y1
            # del y2
            # del loss
            # jt.sync_all(True)
            print(jt.display_memory_info())
            iter_end = time.time()
            print("iter time cost:",iter_end-iter_start)
        end = time.time()
        print("time cost:",end-start)

    def test_memory2(self):
        # return
        start = time.time()
        dim=784
        if True:
            func = MultiInvFunc2(dim)
        else:
            func = MultiForwardFunc2(dim)
        SGD = nn.SGD(func.parameters(), 1e-3)
        for i in range(10):
            print(i)
            x = jt.ones([8192,dim])
            log_det_jacobian = jt.zeros(1)
            y, _ = func(x, log_det_jacobian)
            loss = -jt.mean(y)
            SGD.step(loss)
            # dx = jt.grad(y, func.parameters())
            # del x
            # del y
            # while len(dx)>1:
            #     del dx[len(dx)-1]
            # jt.sync_all(True)
            print(jt.display_memory_info())
        end = time.time()
        print("time cost:",end-start)

if __name__ == "__main__":
    unittest.main()
