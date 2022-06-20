# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
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
    if str(x.dtype).startswith("float"):
        return str(y.dtype).startswith("float") and x.shape == y.shape and (x==y).all()
    return x.dtype == y.dtype and x.shape == y.shape and np.testing.assert_allclose(x, y)

def check(op, *args):
    x = eval(f"np.{op}(*args)")
    y = eval(f"jt.{op}(*args).data")
    all_eq(x, y)

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
            if isinstance(a, list):
                a = np.array(a)
                b = np.array(b)
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
            
            all_eq(ja, a)
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
        
        check("+", [5.0,6.0], [2.0,3.0])
        check("-", [5.0,6.0], [2.0,3.0])
        check("*", [5.0,6.0], [2.0,3.0])
        check("/", [5.0,6.0], [2.0,3.0])
        check("//", [5.0,6.0], [2.0,3.0])
        check("@", [[5,6],[7,8]], [[2,3],[4,5]])
        check("%", [5.0,6.0], [2.0,3.0])
        check("**", [5.0,6.0], [2.0,3.0])
        
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
            
            all_eq(jc, a)
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
        tol = 1e-2 if jt.flags.amp_reg & 2 else 1e-4
        for op in ops:
            func = lambda x: eval(f"((x[0]{op}x[1])*x[2]).sum()")
            x, grads = ngrad(func, [a,b,c], 1e-8)
            ja = jt.array(a).name("ja")
            jb = jt.array(b).name("jb")
            jc = jt.array(c).name("jc")
            jx = eval(f"(ja{op}jb)*jc")
            jgrads = jt.grad(jx, [ja,jb,jc])
            for jd, nd in zip(jgrads, grads):
                np.testing.assert_allclose(jd.data, nd, atol=tol, rtol=tol)

    def test_mod_float(self):
        a = jt.random((10,))
        b = jt.random((10,))
        c = a % b
        assert np.allclose(c.data, a.data % b.data)
        a = jt.random((10,), 'float64')
        b = jt.random((10,), 'float64')
        c = a % b
        assert np.allclose(c.data, a.data % b.data, a.data, b.data)
        if jt.flags.amp_reg & 2: return
        a = jt.random((10,)) * 1000
        b = (jt.random((10,)) * 10).int() + 1
        c = a % b
        assert np.allclose(c.data, a.data % b.data), (c.data, a.data%b.data)

    def test_mod_grad(self):
        a = jt.random((100,))
        b = jt.random((100,))
        c = a % b
        da, db = jt.grad(c, [a, b])
        np.testing.assert_allclose(da.data, 1)
        np.testing.assert_allclose(db.data, -np.floor(a.data/b.data))

    def test_mod_negtive(self):
        a = jt.random((100,)) - 0.5
        b = jt.random((100,)) - 0.5
        c = a % b
        nc = a.data % b.data
        np.testing.assert_allclose(c.data, nc.data, atol=1e-5, rtol=1e-5)
    
    def test_pow(self):
        # win cuda 10.2 cannot pass
        a = jt.random((100,))
        b = a**3
        b.sync()

    def test_binary_op_bool(self):
        a = np.array([0,1,0,1]).astype(bool)
        b = np.array([0,1,1,0]).astype(bool)
        c = np.array([1,1,1,1]).astype(bool)
        check("add", a, b)
        all_eq(np.logical_xor(a, b), jt.subtract(a, b).data)
        check("multiply", a, b)
        check("logical_and", a, b)
        check("logical_or", a, b)
        check("logical_xor", a, b)
        check("bitwise_and", a, b)
        check("bitwise_or", a, b)
        check("bitwise_xor", a, b)
        check("divide", a, c)
        check("floor_divide", a, c)
        check("mod", a, c)


class TestBinaryOpCuda(TestBinaryOp, test_cuda(2)):
    pass

class TestBinaryOpCpuFp16(TestBinaryOp):
    def setUp(self):
        jt.flags.amp_reg = 2 | 4 | 8 | 16
    def tearDown(self):
        jt.flags.amp_reg = 0

@unittest.skipIf(not jt.has_cuda, "no cuda found")
class TestBinaryOpCudaFp16(TestBinaryOp):
    def setUp(self):
        jt.flags.amp_reg = 2 | 4 | 8 | 16
        jt.flags.use_cuda = 1
    def tearDown(self):
        jt.flags.amp_reg = 0
        jt.flags.use_cuda = 0

if __name__ == "__main__":
    unittest.main()