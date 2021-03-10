# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from .test_grad import ngrad
from .test_cuda import test_cuda

def check(op, *args):
    x = eval(f"np.{op}(*args)")
    y = eval(f"jt.{op}(*args).data")
    convert = lambda x: x.astype("uint8") if x.dtype=="bool" else x
    x = convert(x)
    y = convert(y)
    # str match nan and inf
    assert x.dtype == y.dtype and x.shape == y.shape
    for a,b in zip(x.flatten(), y.flatten()):
        assert str(a)[:5] == str(b)[:5], (a,b)

class TestUnaryOp(unittest.TestCase):
    def test_unary_op(self):
        assert jt.float64(1).data.dtype == "float64"
        assert (jt.abs(-1) == 1).data.all()
        assert (abs(-jt.float64(1)) == 1).data.all()
        a = np.array([-1,2,3,0])
        check("abs", a)
        check("negative", a)
        check("logical_not", a)
        check("bitwise_not", a)
        b = np.array([1.1, 2.2, 3.3, 4.4, -1, 0])
        check("log", a)
        check("exp", a)
        check("sqrt", a)
        
    def test_grad(self):
        ops = ["abs", "negative", "log", "exp", "sqrt",
            "sin", "arcsin", "sinh", "arcsinh", 
            "tan", "arctan", "tanh", "arctanh", 
            "cos", "arccos", "cosh", "arccosh", 
            "sigmoid", 
        ]
        a = np.array([1.1, 2.2, 3.3, 4.4])
        for op in ops:
            if op == "abs":
                b = np.array(a+[-1,])
            elif op == "arccosh":
                b = np.array(a)
            elif "sin" in op or "cos" in op or "tan" in op:
                b = np.array(a) / 5
            else:
                b = np.array(a)
            func = lambda x: eval(f"np.{op}(x[0]).sum()")
            if op == "sigmoid":
                func = lambda x: (1/(1+np.exp(-x[0]))).sum()
            x, (da,) = ngrad(func, [b], 1e-8)
            ja = jt.array(b)
            jb = eval(f"jt.{op}(ja)")
            jda = jt.grad(jb, ja)
            assert (np.allclose(jda.data, da)), (jda.data,da,op)

    def test_sigmoid(self):
        a = np.arange(-150,150, 10).astype("float32")
        # a = np.array([-150.0, -140.0, -130.0]).astype("float32")
        b = jt.array(a, dtype='float32')
        b1 = b.sigmoid().numpy()
        assert np.isnan(b1).any() == False

class TestUnaryOpCuda(TestUnaryOp, test_cuda(2)):
    pass

if __name__ == "__main__":
    unittest.main()