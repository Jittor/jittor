# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import sys
import os
import jittor as jt
import unittest
import time
import numpy as np

def get_init_var(shape, dtype):
    return jt.random(shape, dtype)

def batch_norm(x):
    xmean = jt.mean(x, dims=[0,2,3], keepdims=1)
    x2mean = jt.mean(x*x, dims=[0,2,3], keepdims=1)
    norm_x = (x-xmean.broadcast_var(x))/(jt.sqrt(x2mean-xmean*xmean+jt.float32(1e-5)).broadcast_var(x))
    w = jt.make_var([x.shape[1]], init=get_init_var)
    b = jt.make_var([x.shape[1]], init=get_init_var)
    w = w.broadcast([1, w.shape[0],1,1], [0,2,3])
    b = b.broadcast([1, b.shape[0],1,1], [0,2,3])
    return norm_x * w + b

def pool(x, size, op, padding, stride = 1): # TODO: stride, padding
    N,C,H,W = x.shape
    h = (H+padding*2-size)//stride+1
    w = (W+padding*2-size)//stride+1
    xx = x.reindex([N,C,h,w,size,size], [
        "i0", # Nid
        "i1", # Cid
        f"i2*{stride}-{padding}+i4", # Hid
        f"i3*{stride}-{padding}+i5", # Wid
    ])
    return xx.reindex_reduce(op, [N,C,h,w], [
        "i0", # Nid
        "i1", # Cid
        "i2", # Hid
        "i3", # Wid
    ])

def conv(x, in_planes, out_planes, kernel_size, padding, stride = 1):
    Kw = kernel_size
    Kh = kernel_size
    _C = in_planes
    Kc = out_planes
    N,C,H,W = x.shape
    assert C==_C
    w = jt.make_var([Kc, _C, Kh, Kw], init=get_init_var)
    xx = x.reindex([N,Kc,C,(H+padding*2-kernel_size)//stride+1,(W+padding*2-kernel_size)//stride+1,Kh,Kw], [
        'i0', # Nid
        'i2', # Cid
        f'i3*{stride}-{padding}+i5', # Hid+Khid
        f'i4*{stride}-{padding}+i6', # Wid+KWid
    ])
    ww = w.broadcast(xx.shape, [0,3,4])
    yy = xx*ww
    y = yy.sum([2,5,6]) # Kc, Kh, Kw
    return y

def relu(x): return jt.maximum(x, jt.float32(0))

@jt.var_scope('resnet_fake', unique=True)
def resnet_fake(x):
    x = conv(x, 3, 64, 7, 3, 2)
    x = batch_norm(x)
    x = relu(x)
    x = pool(x, 3, "maximum", 1, 2)
    return x

class TestLongestDisFuse(unittest.TestCase):
        
    def test_longest_dis_fuse(self):
        x = jt.array(np.random.rand(1,3,224,224).astype(np.float32))
        loss = jt.sum(resnet_fake(x))
        ps = jt.find_vars('resnet_fake') 
        gs = jt.grad(loss, ps)
        jt.sync(gs)
        # assert not alloc big tensor
        g = jt.dump_all_graphs()
        for s in g.nodes_info:
            if not s.startswith("Var"):
                continue
            shape = s.split("[")[1].split("]")[0].split(",")
            ptr = s.split("(")[1].split(")")[0].split(",")[-1]
            if ptr != '0':
                assert len(shape)<=5, s

if __name__ == "__main__":
    unittest.main()
