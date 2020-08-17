# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error
from .test_grad import ngrad
from .test_cuda import test_cuda

def all_eq(x, y):
    if len(x.shape) == 0: x = np.array([x])
    if len(y.shape) == 0: y = np.array([y])
    convert = lambda x: x.astype("uint8") if x.dtype=="bool" else x
    x = convert(x)
    y = convert(y)
    return x.dtype == y.dtype and x.shape == y.shape and (x==y).all()

def check(op, *args):
    x = eval(f"np.{op}(*args)")
    y = eval(f"jt.{op}(*args).data")
    assert all_eq(x, y), f"{x}\n{y}"

class TestBinaryOp(unittest.TestCase):
    def test_binary_op(self):
        assert np.all(jt.binary(1,2,'maximum').data == 2)
        assert np.all(jt.binary([[1,2]],[[3,4]],'add').data == [[4,6]])
        assert np.all(jt.less(1,2).data)
        assert jt.less(1,2).data.dtype == "bool"
        x = (jt.array(1) << jt.array(3)).data
        assert (x == 8).all()
        x = (jt.array(2) ** jt.array(3)).data
        assert (x == 8).all()
        a = np.array([1,2,3])
        b = np.array([7,10,13])
        check("logical_and", a, b)
        check("logical_or", a, b)
        check("logical_xor", a, b)
        check("bitwise_and", a, b)
        check("bitwise_or", a, b)
        check("bitwise_xor", a, b)
        
    def test_i(self):
        def check(op, a, b):
            if jt.flags.use_cuda and op == "@":
                return
            if op=="@":
                a = np.float32(a)
                b = np.float32(b)
            ja = jt.array(a)
            jb = jt.array(b)
            exec(f"ja {op}= jb")
            ja = ja.fetch_sync()
            
            if op == "@":
                # numpy do not support @=
                a = np.array(a) @ np.array(b)
            else:
                a = eval(f"a {op} b")
                a = np.float32(a)
            ja = np.float32(ja)
            
            assert all_eq(ja, a), (ja,a)
        check("+", 5, 2)
        check("-", 5, 2)
        check("*", 5, 2)
        check("/", 5, 2)
        check("//", 5, 2)
        check("@", [[5]], [[2]])
        check("%", 5, 2)
        check("**", 5, 2)
        check("<<", 5, 2)
        check(">>", 5, 2)
        check("&", 5, 2)
        check("^", 5, 2)
        check("|", 5, 2)
        
    def test_r(self):
        def check(op, a, b):
            a = np.array(a)
            b = np.array(b)
            if jt.flags.use_cuda and op == "@":
                return
            jb = jt.array(b)
            jc = eval(f"a {op} jb").data

            
            if op == "@":
                # numpy do not support @=
                a = np.array(a) @ np.array(b)
            else:
                a = eval(f"a {op} b")
                a = np.array(a)
            
            assert all_eq(jc, a), f"\n{jc}\n{a}"
        check("+", 5, 2)
        check("-", 5, 2)
        check("*", 5, 2)
        check("/", 5, 2)
        check("//", 5, 2)
        # check("@", [[5]], [[2]])
        check("%", 5, 2)
        check("**", 5, 2)
        check("<<", 5, 2)
        check(">>", 5, 2)
        check("&", 5, 2)
        check("^", 5, 2)
        check("|", 5, 2)
        
    def test_grad(self):
        ops = ["+", "-", "*", "/", "**"]
        np.random.seed(3)
        a = np.random.rand(10)
        b = np.random.rand(10)
        c = np.random.rand(10)
        for op in ops:
            func = lambda x: eval(f"((x[0]{op}x[1])*x[2]).sum()")
            x, grads = ngrad(func, [a,b,c], 1e-8)
            ja = jt.array(a).name("ja")
            jb = jt.array(b).name("jb")
            jc = jt.array(c).name("jc")
            jx = eval(f"(ja{op}jb)*jc")
            jgrads = jt.grad(jx, [ja,jb,jc])
            for jd, nd in zip(jgrads, grads):
                assert (np.abs(jd.data-nd)<1e-4).all(), f"\n{jd.data}\n{nd}"

    def test_mod_float(self):
        a = jt.random((10,))
        b = jt.random((10,))
        c = a % b
        assert np.allclose(c.data, a.data % b.data)
        a = jt.random((10,), 'float64')
        b = jt.random((10,), 'float64')
        c = a % b
        assert np.allclose(c.data, a.data % b.data)
        a = jt.random((10,)) * 1000
        b = (jt.random((10,)) * 10).int() + 1
        c = a % b
        assert np.allclose(c.data, a.data % b.data), (c.data, a.data%b.data)


class TestBinaryOpCuda(TestBinaryOp, test_cuda(2)):
    pass

if __name__ == "__main__":
    unittest.main()