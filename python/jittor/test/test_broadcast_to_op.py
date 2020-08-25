# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error
from .test_cuda import test_cuda
import contextlib

def gen_data(shape):
    num = np.multiply.reduce(shape)
    a = np.arange(0, num)
    return a.reshape(shape)

class TestBroadcastToOp(unittest.TestCase):
    def setUp(self):
        self.use_shape = False
        
    def test1(self):
        def check(shape1, shape2):
            a = gen_data(shape1)
            b = gen_data(shape2)
            aa,bb = np.broadcast_arrays(a, b)
            if self.use_shape:
                ja = jt.ops.broadcast(a, shape2).data
            else:
                ja = jt.ops.broadcast_var(a, b).data
            assert ja.shape == aa.shape and (ja==aa).all(), f"{ja}, {aa}"
        check([1], [3])
        check([3,1], [3])
        check([3,1,3], [1,3,1])
        check([2,3,4], [2,3,4,1,1,1])
        check([2,3], [2,3,1,1])
        check([2,1,3,1,4], [1,3,4])
        
        expect_error(lambda: jt.ops.broadcast_var([1,2],[1,2,3]))
        
    def test_binary_op(self):
        if self.use_shape: return
        def check(shape1, shape2):
            a = gen_data(shape1)
            b = gen_data(shape2)
            x = y = None
            try:
                x = a+b
            except Exception as e:
                pass
            try:
                y = jt.ops.add(a, b).data
            except Exception as e:
                pass
            assert (x==y).all(), f"{x}\n{y}"
        check([1], [3])
        check([3,1], [3])
        check([3,1,3], [1,3,1])
        check([2,3,4], [2,3,4,1,1,1])
        check([2,3], [2,3,1,1])
        check([2,1,3,1,4], [1,3,4])

class TestBroadcastToOpForward(unittest.TestCase):
    def test_forward(self):
        @contextlib.contextmanager
        def check(bop_num):
            jt.clean()
            yield
            graph = jt.dump_all_graphs()
            bop = [ node for node in graph.nodes_info 
                if node.startswith("Op") and "broadcast_to" in node]
            assert len(bop)==bop_num, (len(bop), bop_num)
        
        with check(1):
            a = jt.array([1,2,3])
            b = a+1
        assert (b.data==[2,3,4]).all()
        del a, b

        with check(0):
            a = jt.array([1,2,3])
            b = a+a
        assert (b.data==[2,4,6]).all()
        del a, b

        def test_shape(shape1, shape2, bop_num):
            with check(bop_num):
                a = jt.random(shape1)
                b = jt.random(shape2)
                c = a+b
        test_shape([3,3,3], [3,3,3], 0)
        test_shape([3,3,3], [3,3,1], 1)
        test_shape([3,3,3], [3,1,1], 1)
        test_shape([3,3,3], [1,1,1], 1)
        test_shape([3,3,3], [1,1,3], 1)
        test_shape([3,3,3], [1,3,3], 1)
        test_shape([3,3,1], [1,3,3], 2)
        test_shape([3,1,3], [1,3,3], 2)
        test_shape([3,3], [1,3,3], 1)
        test_shape([3,3], [1,3,1], 2)


class TestBroadcastToOp2(TestBroadcastToOp):
    def setUp(self):
        self.use_shape = True

@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestBroadcastToOpCuda(TestBroadcastToOp):
    def setUp(self):
        jt.flags.use_cuda = 2
        self.use_shape = False
    def tearDown(self):
        jt.flags.use_cuda = 0

@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestBroadcastToOp2Cuda(TestBroadcastToOp):
    def setUp(self):
        jt.flags.use_cuda = 2
        self.use_shape = True
    def tearDown(self):
        jt.flags.use_cuda = 0


class TestBroadcastToOpMisc(unittest.TestCase):
    def test_negtive_dim(self):
        a = jt.array([1,2])
        assert (a.broadcast([2,2], [-1]).data == [[1,1],[2,2]]).all()
        assert (a.broadcast([2,2], [-2]).data == [[1,2],[1,2]]).all()
        
    def test_negtive_dim2(self):
        a = jt.array([1,2])
        b = jt.zeros((2,2))
        assert (a.broadcast(b, [-1]).data == [[1,1],[2,2]]).all()
        assert (a.broadcast(b, [-2]).data == [[1,2],[1,2]]).all()

    def test_zero_dim(self):
        a = jt.array(1.0)
        b = a.broadcast([0])
        assert b.shape == [0]


if __name__ == "__main__":
    unittest.main()