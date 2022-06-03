# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import random
from .test_log import find_log_with_re
from .test_core import expect_error

def plus(a, b):
    return a + b

def subtraction(a, b):
    return a - b

def multiplication(a, b):
    return a * b

def division(a, b):
    return a / b

def get_random_op():
    v = random.randint(0, 3)
    if (v == 0):
        return plus
    elif (v == 1):
        return subtraction
    elif (v == 2):
        return multiplication
    else:
        return division
    
def test(shape, op1, op2):
    n = 753.1
    a = jt.random(shape)
    b = jt.random(shape)
    c = op1(a, n)
    d = op2(c, b)
    with jt.log_capture_scope(log_v=0, log_vprefix="fused_op.cc=100") as logs:
        d__ = d.data
    logs = find_log_with_re(logs, 
        "Jit (fused )?op key (not )?found: «opkey0:array«T:float32")
    assert(len(logs)==1), logs

    a_ = a.data
    b_ = b.data
    d_ = op2(op1(a_, n), b_)
    assert(np.allclose(d_, d__, atol=1e-4))

def gen_data(shape):
    num = np.multiply.reduce(shape)
    a = np.arange(0, num)
    return a.reshape(shape)

class TestSingleArray(unittest.TestCase):
    def test7(self):
        a = jt.random([100])
        x = a.reindex_var((a>0.1).where())
        x.data

    def test6(self):
        jt.clean()
        def check(hv, lv, lo):
            self.assertEqual(jt.number_of_hold_vars(), hv)
            self.assertEqual(jt.number_of_lived_vars(), lv)
            self.assertEqual(jt.number_of_lived_ops(), lo)
        check(0,0,0)
        a = jt.array(1.0).name('a').stop_fuse()
        b = (a+jt.array(1.0).name('t1').stop_fuse()).name('b')
        c = (b+jt.array(1.0).name('t2').stop_fuse()).name('c')
        check(3,5,5)
        graph = jt.dump_all_graphs()
        self.assertEqual(c.data, 3)
        check(3,5,2)
    
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test5(self):
        with jt.flag_scope(use_cuda=1):
            f32 = jt.float32
            np.random.seed(0)
            jt.set_seed(3)

            x = f32(np.random.rand(1, 1))
            w = (jt.random([x.shape[-1], 10])-f32(0.5)) / f32(x.shape[-1])**f32(0.5)
            jt.nn.matmul(x, w).data

    def test4(self):
        jt.array(1).data

    def test_concat(self):
        def check(shape, dim, n):
            num = np.prod(shape)
            arr1 = []
            arr2 = []
            for i in range(n):
                a = (np.array(range(num)) + i*num).reshape(shape)
                arr1.append(a)
                arr2.append(jt.array(a))
            x = np.concatenate(tuple(arr1), dim)
            y = jt.concat(arr2, dim)
            assert (x==y.data).all()
        check([1], 0, 20)

    def test3(self):
        def check(shape1, shape2):
            a = gen_data(shape1)
            b = gen_data(shape2)
            aa,bb = np.broadcast_arrays(a, b)
            ja = jt.ops.broadcast_var(a, b).data
            assert ja.shape == aa.shape and (ja==aa).all(), f"{ja}, {aa}"
        check([1], [3])

    def test2(self):
        a = jt.random([5])
        a = a * 2000 - 1000
        a.data

    def test_main(self):
        test_n = 10
        test([50, 50, 50, 50], multiplication, subtraction)
        for i in range(test_n):
            n = random.randint(1,4)
            shape = []
            for j in range(n):
                shape.append(random.randint(1,50))
            test(shape, get_random_op(), get_random_op())

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test_main_cuda(self):
        with jt.flag_scope(use_cuda=1):
            test_n = 10
            test([50, 50, 50, 50], multiplication, subtraction)
            for i in range(test_n):
                n = random.randint(1,4)
                shape = []
                for j in range(n):
                    shape.append(random.randint(1,50))
                test(shape, get_random_op(), get_random_op())

if __name__ == "__main__":
    unittest.main()