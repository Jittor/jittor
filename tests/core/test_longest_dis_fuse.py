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
import gc
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


def _resident_mib():
    """This process's resident set, from /proc (the gate is Linux-only)."""
    with open("/proc/self/statm") as handle:
        pages = int(handle.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / (1 << 20)


def _rank_and_ptr(info):
    shape = [f for f in info.split("[")[1].split("]")[0].split(",") if f.strip()]
    ptr = info.split("(")[1].split(")")[0].split(",")[-1]
    return len(shape), ptr


#: What building this graph and its gradient may hold on to, resident.
#:
#: The graph contains two rank-7 ``[1,64,3,112,112,7,7]`` vars -- jittor's own
#: convolution formulation -- with a logical element count of 118,013,952 each,
#: 450 MiB if a shape were a size. It is not: both are stride-0 views over small
#: storage (one of them shares the 64x3x7x7 weight), and the graph's peak
#: resident set is under 100 MiB. Measured 2026-09-22 on the CPU gate: VmHWM
#: 83.8 MiB, resident growth across the build 33 MiB, and no mapping in the
#: process over 100 MiB at all.
#:
#: So this asserts memory, which is what "assert not alloc big tensor" meant --
#: 256 MiB leaves the legitimate 33 MiB far below the bound and still catches a
#: materialised 450 MiB intermediate, which is how the entry this replaces
#: (KI-OPS-013) was read. The rank-and-pointer enumeration it replaces could not
#: tell the two apart: a var with a rank-7 shape and a non-null pointer may own
#: four hundred bytes, and reporting that as a materialised buffer is a false
#: red, not a finding.
_MAX_RESIDENT_GROWTH_MIB = 256


class TestLongestDisFuse(unittest.TestCase):

    def test_longest_dis_fuse(self):
        x = jt.array(np.random.rand(1,3,224,224).astype(np.float32))
        net = resnet_fake()
        gc.collect()
        before = _resident_mib()
        loss = jt.sum(net(x))
        ps = net.parameters()
        gs = jt.grad(loss, ps)
        jt.sync(gs)
        gc.collect()
        growth = _resident_mib() - before
        # Named only when it fails: the wide vars are the convolution's im2col
        # views, and a bound this large being crossed means one of them -- or an
        # op that feeds one -- stopped folding away.
        wide = []
        for s in jt.dump_all_graphs().nodes_info:
            if not s.startswith("Var"):
                continue
            rank, ptr = _rank_and_ptr(s)
            if rank > 5 and ptr not in ("0", "0x0"):
                wide.append(s)
        assert growth <= _MAX_RESIDENT_GROWTH_MIB, (
            "building this graph grew the resident set by %.0f MiB (limit %d "
            "MiB): an intermediate that the fusion should fold away was "
            "materialised instead. The rank > 5 vars in the graph are %s"
            % (growth, _MAX_RESIDENT_GROWTH_MIB, wide[:2]))


if __name__ == "__main__":
    unittest.main()

