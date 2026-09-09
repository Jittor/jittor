# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
from _helpers.numerical_grad import ngrad

def check(op, *args):
    x = eval(f"np.{op}(*args)")
    y = eval(f"jt.{op}(*args).numpy()")

    def convert(value):
        return value.astype("uint8") if value.dtype == "bool" else value

    x = convert(x)
    y = convert(y)
    # str match nan and inf
    assert x.dtype == y.dtype and x.shape == y.shape, \
        (x.dtype, y.dtype, x.shape, y.shape)
    for a,b in zip(x.flatten(), y.flatten()):
        assert str(a)[:5] == str(b)[:5], (a,b)

class UnaryOpCases:
    __test__ = False

    def test_unary_op(self):
        assert jt.float64(1).data.dtype == "float64"
        assert (jt.abs(-1) == 1).data.all()
        assert (abs(-jt.float64(1)) == 1).data.all()
        a = np.array([-1,2,3,0], dtype="int32")
        check("abs", a)
        check("negative", a)
        check("logical_not", a)
        check("bitwise_not", a)
        dtype = "float16" if (jt.introspection.policy.runtime.amp_reg & 2) else "float32"
        check("log", a.astype(dtype))
        check("exp", a.astype(dtype))
        check("sqrt", a.astype(dtype))
        
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
            def func(values):
                return eval(f"np.{op}(values[0]).sum()")

            if op == "sigmoid":
                def func(values):
                    return (1 / (1 + np.exp(-values[0]))).sum()
            x, (da,) = ngrad(func, [b], 1e-8)
            ja = jt.array(b)
            jb = eval(f"jt.{op}(ja)")
            jda = jt.grad(jb, ja)
            tol = 1e-2 if jt.introspection.policy.runtime.amp_reg & 2 else 1e-6
            assert (np.allclose(jda.data, da, atol=tol, rtol=tol)), (jda.data,da,op)

    def test_sigmoid(self):
        a = np.arange(-150,150, 10).astype("float32")
        # a = np.array([-150.0, -140.0, -130.0]).astype("float32")
        b = jt.array(a, dtype='float32')
        b1 = b.sigmoid().numpy()
        assert not np.isnan(b1).any()

    def test_safe_clip(self):
        a = jt.array([-1.0,0,0.4,1,2,3])
        b = a.safe_clip(0.1, 0.5)
        tol = 1e-3 if jt.introspection.policy.runtime.amp_reg & 2 else 1e-6
        np.testing.assert_allclose(
            b.data, [0.1,0.1,0.4,0.5,0.5,0.5], atol=tol, rtol=tol
        )
        da = jt.grad(b, a)
        assert (da.data == 1).all()

    def test_erfinv(self):
        from scipy import special
        y = np.linspace(-1.0, 1.0, num=10)
        x = special.erfinv(y)
        y2 = jt.array(y)
        x2 = jt.erfinv(y2)
        tol = 1e-3 if jt.introspection.policy.runtime.amp_reg & 2 else 1e-6
        np.testing.assert_allclose(x2.data, x, atol=tol, rtol=tol)

        y = np.linspace(-0.9, 0.9, num=10)
        x = special.erfinv(y)
        y2 = jt.array(y)
        x2 = jt.erfinv(y2)
        np.testing.assert_allclose(x2.data, x, atol=tol, rtol=tol)
        d = jt.grad(x2, y2)
        _, (dn,) = ngrad(lambda y: special.erfinv(y).sum(), [y], 1e-8)
        np.testing.assert_allclose(d.data, dn, atol=tol, rtol=tol)
