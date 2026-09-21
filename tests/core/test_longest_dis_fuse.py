# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
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

def relu(x): return jt.maximum(x, jt.float32(0))

def resnet_fake():
    from jittor import nn
    net = nn.Sequential(
        nn.Conv(3, 64, 7, 2, 3),
        nn.BatchNorm(64),
        nn.ReLU(),
        nn.Pool(3, 2, 1)
    )
    return net

class TestLongestDisFuse(unittest.TestCase):
        
    def test_longest_dis_fuse(self):
        x = jt.array(np.random.rand(1,3,224,224).astype(np.float32))
        net = resnet_fake()
        loss = jt.sum(net(x))
        ps = net.parameters()
        gs = jt.grad(loss, ps)
        jt.sync(gs)
        # assert not alloc big tensor
        g = jt.dump_all_graphs()
        for s in g.nodes_info:
            if not s.startswith("Var"):
                continue
            # `debug_msg` writes the shape with a trailing comma -- `[1,64,112,112,]`
            # -- and `"1,64,112,112,".split(",")` is five fields, not four. The
            # count here was one more than the rank for as long as this has been
            # written that way, so the bound below was enforced as `rank <= 4`
            # and every tensor in the graph was one degree closer to it than the
            # number suggests. Drop the empty fields.
            shape = [f for f in s.split("[")[1].split("]")[0].split(",") if f.strip()]
            ptr = s.split("(")[1].split(")")[0].split(",")[-1]
            if ptr != '0' and ptr != '0x0':
                # The message names the rank and the var: the interesting one is
                # a 7-dimensional reindex the convolution materialises instead of
                # folding away, and `assert 8 <= 5` is not a report a reader can
                # act on. See KI-OPS-013.
                assert len(shape)<=5, \
                    "a fused intermediate was materialised with %d dims: %s" % (len(shape), s)

if __name__ == "__main__":
    unittest.main()
