# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import unittest
import time
import jittor as jt
from jittor import LOG
import numpy as np
from .test_core import expect_error
import contextlib

performance_test = os.environ.get("performance_test", "") == "1"
skip_slow_test = not performance_test

@contextlib.contextmanager
def performance_test_scope(warmup=0, rerun=0, **args):
    """ profile scope
    example:
        with jt.profile_scope() as report:
            ......
        print(report)
    """
    assert not jt.flags.profiler_enable
    if skip_slow_test:
        jt.profiler.start(0, 0)
    else:
        jt.profiler.start(warmup, rerun)
    report = []
    try:
        with jt.flag_scope(**args):
            yield report
    finally:
        jt.profiler.stop()
        if skip_slow_test:
            report.extend([[1e30]]*3)
        else:
            report.extend(jt.profiler.report())

def retry(num):
    def outer(func):
        def inner(*args):
            for i in range(num):
                if i == num-1:
                    func(*args)
                    break
                try:
                    func(*args)
                    break
                except:
                    pass
                LOG.v(f"Retry {i}")
        return inner
    return outer

def get_np_matmul_toughtput(size):
    # import os
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # import numpy as np
    # import time
    a = np.random.randn(size, size).astype("float32")
    b = np.random.randn(size, size).astype("float32")
    c = np.random.randn(size, size).astype("float32")
    warmup = 2
    rerun = 10+1
    for _ in range(warmup): np.matmul(a,b,c)
    start_time = time.time()
    for _ in range(rerun): np.matmul(a,b,c)
    elapsed_time = time.time() - start_time
    return (size*size*size*rerun) / elapsed_time

class TestFusedOp(unittest.TestCase):
    def test_add(self):
        jt.clean()
        def check(hv, lv, lo):
            self.assertEqual(jt.number_of_hold_vars(), hv)
            self.assertEqual(jt.number_of_lived_vars(), lv)
            self.assertEqual(jt.number_of_lived_ops(), lo)
        for i in range(8):
            check(0,0,0)
            a = jt.array(1.0).name('a').stop_fuse()
            b = (a+jt.array(1.0).name('t1').stop_fuse()).name('b')
            c = (b+jt.array(1.0).name('t2').stop_fuse()).name('c')
            check(3,5,5)
            graph = jt.dump_all_graphs()
            self.assertEqual(c.data, 3)
            check(3,5,2)
            graph = jt.dump_all_graphs()
            for node in graph.nodes_info:
                if node.startswith("Op"):
                    if 'add->' in node:
                        assert ':s0' in node, node
                    else:
                        assert ':s1' in node, node
                elif ',b,' in node:
                    # b has been fused
                    assert ':s0' in node, node
                else:
                    assert ':s1' in node
            if i&1: del a
            if i&2: del b
            if i&4: del c
            
            if i==0: check(3,5,2)
            elif i==1: check(2,5,2)
            elif i==2: check(2,5,2)
            elif i==3: check(1,1,0)
            elif i==4: check(2,3,1)
            elif i==5: check(1,3,1)
            elif i==6: check(1,1,0)
            elif i==7: check(0,0,0)
            
            if not (i&1): a.sync()
            if not (i&2): b.sync()
            if not (i&4): c.sync()
            
            if i==0: check(3,5,2)
            elif i==1: check(2,3,1)
            elif i==2: check(2,5,2)
            elif i==3: check(1,1,0)
            elif i==4: check(2,3,1)
            elif i==5: check(1,1,0)
            elif i==6: check(1,1,0)
            
            if not (i&1): del a
            if not (i&2): del b
            if not (i&4): del c
            check(0,0,0)
            
    def test_fuse_reduce_and_broadcast(self):
        size = 10
        a = jt.random([size,size,1])
        b = jt.random([1,size,size])
        c = (a*b).sum(dim=1)
        nc = (a.data*b.data).sum(1)
        assert c.shape == [size,size]
        assert (np.abs(nc-c.data)<1e-5).all()
        
    def test_fuse_reduce_and_broadcast2(self):
        size = 10
        a = jt.random([1])
        b = a.broadcast([size]).sum()
        assert (np.abs(b.data - a.data*size) < 1e-5).all()
        a = jt.random([size,1])
        b = a.broadcast([size,size]).sum(1, keepdims=True)
        assert (np.abs(b.data - a.data*size) < 1e-5).all()
    
    def test_fuse_reduce2(self):
        size = 10
        a = jt.random([1]).broadcast([size]).name('a')
        # a.data
        b = a.sum().name('b')
        c = a.min().name('c')
        d = a.max().name('d')
        jt.fetch_sync([b,c,d])
        
        graph = jt.dump_all_graphs()
        node_a = [ node for node in graph.nodes_info if ",a," in node ]
        assert 's0' in node_a[0]
        
        v = a.data[0]
        assert np.allclose(v*10,b.data) and v==c.data and v==d.data, (v, b.data, c.data, d.data)
        
    def test_profile_fused_op(self):
        size = 1000
        r1 = []
        r2 = []
        for size in range(1024, 1025, 1):
            with performance_test_scope(2, 10) as report:
                a = jt.random([size,size,1])
                b = jt.random([1,size,size])
                c = (a*b).sum(1)
                c.sync()
            
            assert len(report) == 3
            tp_np = get_np_matmul_toughtput(size)
            tp_jt = float(report[1][-1])
            r1.append(tp_jt)
            r2.append(tp_np)
            na = a.data.reshape((size,size))
            nb = b.data.reshape((size,size))
            nc = np.matmul(na,nb)
            assert (np.abs(nc-c.data)<1e-2).all(), np.abs(nc-c.data).max()
    
    # @unittest.skipIf(skip_slow_test, "Skip slow test")
    def test_profile_fused_op_transpose(self):
        for size in range(1024, 1025, 1):
            with performance_test_scope(2, 10):
                b = jt.random([size,1,size])
                a = jt.random([1,size,size])
                c = (a*b).sum(2)
                c.data
        
    # @unittest.skipIf(skip_slow_test, "Skip slow test")
    def test_profile_fused_op_split(self):
        # match v4
        @retry(10)
        def check(n, m, k, cs, rs, rtp):
            a = jt.random([n,m,1])
            b = jt.random([1,m,k])
            a.data, b.data
            with performance_test_scope(
                20, 20000000000//(n*m*k),
                compile_options = {
                "split0":16,"split1":6,"split2":16,
                "order0":0, "order1":1,"order2":1,
                "order3":0, "order4":0,"order5":0,
                "restride":rs, "unroll":2, "vectorize":2,
                "compile_shapes":cs
            }) as report:
                c = (a*b).sum(1)
            c.data
            
            na = a.data.reshape((n,m))
            nb = b.data.reshape((m,k))
            nc = np.matmul(na,nb)
            
            assert (np.abs(nc-c.data)<1e-2).all(), np.abs(nc-c.data).max()
            tp = float(report[-1][-1])
            assert tp > rtp * 10**9, (tp, rtp)

        check(65, 8, 19, 1, 0, 0)
        check(64, 6, 16, 1, 0, 33) # TODO 36
        check(64, 6, 16, 0, 0, 21)
        check(64, 60, 16, 1, 0, 44)
        check(64, 60, 16, 0, 0, 30)
        check(65, 60, 16, 0, 0, 30)
        check(65, 61, 16, 0, 0, 27)
        check(65, 65, 16, 0, 0, 26)
        check(64, 60, 64, 1, 1, 27)
        check(64, 60, 64, 1, 0, 42)
        check(64, 60, 64, 0, 0, 30) # TODO: why slower?

    
    @unittest.skipIf(skip_slow_test, "Skip slow test")
    def test_profile_fused_op_restride(self):
        # match v6

        @retry(10)
        def check(n, m, k, cs, rs, pa, rtp):
            a = jt.random([n,m,1])
            b = jt.random([1,m,k])
            a.data, b.data
            with performance_test_scope(
                0, 20000000000//(n*m*k),
                compile_options = {
                "order0":0, "order1":0,"order2":0,
                "split0":64,"split1":60,"split2":64,
                "split3":16,"split4":6,"split5":16,
                "order3":0, "order4":1,"order5":1,
                "order6":0, "order7":0,"order8":0,
                "restride":rs,"vectorize":2,"unroll":2,
                "compile_shapes":cs, "parallel":pa
            }) as report:
                c = (a*b).sum(1)
            c.sync()

            na = a.data.reshape((n,m))
            nb = b.data.reshape((m,k))
            nc = np.matmul(na,nb)
            assert (np.abs(nc-c.data)/nc<1e-5).all(), (np.abs(nc-c.data).max(), np.where(np.abs(nc-c.data)>1))
            tp = float(report[-1][-1])
            assert tp > rtp * 10**9, (tp, rtp)
            
        check(64*1, 60*1, 64*1, 0, 0, 0, 31)
        check(64*1, 60*1, 64*1, 0, 1, 0, 25)
        check(64*1, 60*1, 64*1, 1, 0, 0, 42)
        check(64*55, 60*55, 64*55, 0, 0, 0, 20)
        check(64*55, 60*55, 64*55, 1, 1, 0, 37)
        check(64*55, 60*55, 64*55, 0, 1, 0, 36)
        check(64*55+1, 60*55+1, 64*55, 0, 1, 0, 36)
        check(64*55+1, 60*55+1, 64*55+1, 0, 1, 0, 36)
        check(64*55+15, 60*55+15, 64*55+15, 0, 1, 0, 34) # TODO: 36
        check(64*16, 60*16, 64*16, 0, 1, 0, 35)
        check(64*55, 60*55, 64*55, 0, 1, 0, 36)

    
    @unittest.skipIf(skip_slow_test, "Skip slow test")
    def test_profile_fused_op_split3(self):
        # match v6

        n, m, k = 64*100, 60*100, 64*100
        
        a = jt.random([n,m,1])
        b = jt.random([1,m,k])
        # a = jt.ones([n,m,1])
        # b = jt.ones([1,m,k])
        a.data, b.data
        with performance_test_scope(
            0, 400000000000//(n*m*k),
            compile_options={
            "order0":0, "order1":0,"order2":0,
            "split0":64*4,"split1":60*4,"split2":64*4,
        
            "order3":0, "order4":1,"order5":1,
            "split3":64,"split4":60,"split5":64,
        
            "order6":0, "order7":1,"order8":1,
            "split6":16,"split7":6,"split8":16,
        
            "order9":0, "order10":0,"order11":0,
        
            "restride":1, "unroll":2,"vectorize":2
        }):
            c = (a*b).sum(1)
        c.data

        na = a.data.reshape((n,m))
        nb = b.data.reshape((m,k))
        nc = np.matmul(na,nb)
        
        assert (np.abs(nc-c.data)/nc<1e-5).all(), (np.abs(nc-c.data).max(), np.where(np.abs(nc-c.data)>1))

    @unittest.skipIf(skip_slow_test, "Skip slow test")
    def test_profile_fused_op_parallel(self):
        # match v6

        @retry(10)
        def check(n, m, k, cs, rs, pa, rtp):
            a = jt.random([n,m,1])
            b = jt.random([1,m,k])
            a.data, b.data
            with performance_test_scope(
                2, 30,
                compile_options = {
                "order0":0, "order1":0,"order2":0,
                "split0":64,"split1":60,"split2":64,
                "split3":16,"split4":6,"split5":16,
                "order3":0, "order4":1,"order5":1,
                "order6":0, "order7":0,"order8":0,
                "restride":rs,"vectorize":2,"unroll":2,
                "compile_shapes":cs, "parallel":pa
            }) as report:
                c = (a*b).sum(1)
            c.data

            na = a.data.reshape((n,m))
            nb = b.data.reshape((m,k))
            nc = np.matmul(na,nb)
            
            assert (np.abs(nc-c.data)/nc<1e-5).all(), (np.abs(nc-c.data).max(), np.where(np.abs(nc-c.data)>1))
            tp = float(report[-1][-1])
            assert tp > rtp * 10**9, (tp, rtp)
            
        check(64*16, 60*16, 64*16, 1, 1, 0, 35)
        check(64*16, 60*16, 64*16, 1, 1, 1, 60)
        check(64*16, 60*16, 64*16, 0, 1, 0, 35)
        check(64*16, 60*16, 64*16, 0, 1, 1, 60)
        check(64*16+5, 60*16+5, 64*16+5, 0, 1, 1, 50)

if __name__ == "__main__":
    unittest.main()