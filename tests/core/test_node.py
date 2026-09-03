# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from _helpers.assertions import expect_error
from jittor_utils import LOG
import time, os

#: Counts at the start of the case being checked, so `check` can assert a
#: *difference*.
_BASELINE = (0, 0, 0)


def counts():
    import gc
    gc.collect()
    jt.graph_check()
    return (jt.number_of_hold_vars(), jt.number_of_lived_vars(),
            jt.number_of_lived_ops())


def baseline():
    """Pin what this process already holds; everything below is relative to it."""
    global _BASELINE
    _BASELINE = counts()
    return _BASELINE


def check(hv, lv, lo):
    """How many nodes *this* case created, not how many the process holds.

    The absolute form -- `assert counts() == (0, 0, 0)` after `jt.clean()` --
    is wrong by construction and only looked right because this file used to
    run early in a fresh process. Jittor keeps bounded, process-level caches of
    Vars (`jittor.fft._dft_mat_cache`, `jittor.nn.attention._CU_SEQLENS_CACHE`),
    so the floor depends on which operators the process has already run, not on
    this test. It fails as soon as anything runs first -- which is exactly what
    a parallel gate does, because the worker's file order is not the tree's.
    Measured: red under `-n 4 --dist loadfile`, green alone.

    The property being tested is unchanged and is the interesting one anyway:
    this graph creates two arrays, then a product, then two gradients.
    """
    a, b, c = counts()
    got = (a - _BASELINE[0], b - _BASELINE[1], c - _BASELINE[2])
    assert got == (hv, lv, lo), (got, _BASELINE, (a, b, c),
                                 jt.dump_all_graphs().nodes_info)

def get_xorshf96(seed=0):
    '''Marsaglia's xorshf generator'''
    a = [
        np.uint64(123456789+seed),
        np.uint64(362436069+seed),
        np.uint64(521288629+seed),
    ]
    def xorshf96():
        a[0] ^= a[0] << np.uint64(16)
        a[0] ^= a[0] >> np.uint64(5)
        a[0] ^= a[0] << np.uint64(1)
        t = a[0]
        a[0] = a[1]
        a[1] = a[2]
        a[2] = t ^ a[0] ^ a[1]
        return int(a[2])
    # for _ in range(10): xorshf96()
    return xorshf96

class TestNode(unittest.TestCase):
    def test_lived(self):
        jt.clean()
        baseline()
        check(0,0,0)
        a = jt.array(1.0).stop_fuse()
        a.name('a')
        b = jt.array(1.0).stop_fuse()
        b.name('b')
        check(2,2,2)
        c = a * b
        c.name('c')
        check(3,3,3)
        vc = c.numpy()
        check(3,3,1)
        da, db = jt.grad(c, [a, b])
        da.name('da')
        db.name('db')
        check(5,6,4) # dc, 3, da, 1, db, 1
        del a, b, c
        check(2,5,3)
        da.sync(), db.sync()
        check(2,2,0)
        del da, db
        check(0,0,0)

    def test_pending(self):
        a = jt.float([1,2,3])
        b = jt.float([1,2,3])
        c = a.float().float().float() * b.float().float().float()
        del a
        c.data
        assert (c.data==[1,4,9]).all(), c.data
        d, = jt.grad(c, [b])
        d.data
        assert (d.data==[1,2,3]).all(), d.data

    def test_node_performance(self):
        mode = os.environ.get("test_node_performance")
        if mode==None or mode not in "12":
            return
        if mode=="1":
            bc = lambda x: jt.broadcast(x, [1,1,1,1],[0,1,2])
            rd = lambda x: jt.sum(x)
        else:
            bc = lambda x: jt.reindex(x, [1,1,1,1],["i0+i1+i2+i3"])
            rd = lambda x: jt.reindex_reduce(x, "add", [1], ["i0+i1+i2+i3"])
        if jt.compiler.is_debug: return
        def run():
            start_time = time.time()
            fop_num = 10000
            fop_input_num = (2, 3) # (i,j) -> [i,i+j] -> [2, 5]
            # fop_output_num = (1, 0) # [1,1]
            inner_op_num = (0, 3)
            fop_type_num = 63 # how many different fuse op
            input_queue_num = 15
            queue = [1.0]*(input_queue_num+1)
            x = get_xorshf96()
            rand = lambda x, l, r: l+((x())&r)
            ops = ["add", "subtract", "multiply", "divide"]
            get_op = lambda x: ops[(x())&3]
            for i in range(fop_num):
                prev = bc(queue[rand(x,0,input_queue_num)])
                y = get_xorshf96(x()&fop_type_num)
                inum = rand(y, *fop_input_num)
                q = [prev]
                for i in range(inum-1):
                    n = bc(queue[rand(x,0,input_queue_num)])
                    prev = jt.binary(prev, n, get_op(y))
                    q.append(prev)
                innum = rand(y,*inner_op_num)
                for _ in range(innum):
                    j = rand(y,0,len(q)-1)
                    n = q[j]
                    prev = jt.binary(prev, n, get_op(y))
                    q[j] = prev
                prev = rd(prev)
                queue[rand(x,0,input_queue_num)] = prev
            a = jt.array(0.0)
            for x in queue:
                a += x
            LOG.i("build graph", time.time()-start_time, jt.liveness_info().values())
            start_time = time.time()
            a.sync()
            LOG.i("execute", time.time()-start_time)
        # debug mode:  build(0.68), execute(0.44)
        # normal mode: build(0.56), execute(0.25)
        # cast opt:    build(0.50), execute(0.25)
        # dtype opt:   build(0.49), execute(0.25)
        # pyjt opt:    build(0.48), execute(0.25)
        # ns opt:      build(0.46), execute(0.24)
        # nv opt:      build(0.42), execute(0.23)
        # nv opt:      build(0.415),execute(0.225)
        # jit_key opt: build(0.415),execute(0.15)
        # jit_key opt: build(0.415),execute(0.11)
        # sv opt:      build(0.42), execute(0.12)
        # noded opt:   build(0.42), execute(0.10)

        # tcm opt:     build(0.40), execute(0.10)

        # mode2: reindex
        # jit_key opt: build(0.46),execute(0.12)
        # noded opt:   build(0.44),execute(0.11)
        # for i in range(20):
        #     run()

        # version 1.3.2.6 retest(laptop)
        # mode1:
        # origin      0.296 exec(0.11)
        # int32flag   0.298 exec(0.11)
        # add order   0.299 exec(0.11)
        # rm p1 rule  0.299 exec(0.11)
        for i in range(20):
            run()
        import gc
        gc.collect()
        run()
        

if __name__ == "__main__":
    unittest.main()
