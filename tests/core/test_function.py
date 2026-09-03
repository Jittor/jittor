# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import pytest
from collections.abc import Sequence, Mapping
from _helpers.assertions import expect_error
from _helpers.state_leaks import assert_rss_growth_bounded
from jittor import Function

class TestFunction(unittest.TestCase):
    def test_first_order_only_gradient_rejects_higher_order(self):
        class FirstOrderOnlySquare(Function):
            def execute(self, value):
                self.value = value
                return value * value

            def grad(self, grad):
                result = 2 * self.value * grad
                result._set_first_order_only()
                return result

        value = jt.array([2.0, 3.0])
        first = jt.grad(FirstOrderOnlySquare()(value).sum(), value)
        np.testing.assert_allclose(first.numpy(), [4.0, 6.0])
        with self.assertRaisesRegex(
                RuntimeError, "Higher-order gradients.*first-order-only"):
            jt.grad(first.sum(), value)

        next_value = jt.array([2.0, 3.0])
        next_first = jt.grad(
            FirstOrderOnlySquare()(next_value).sum(), next_value)
        next_first.stop_grad()
        updated = next_value - 0.1 * next_first
        next_gradient = jt.grad((updated * updated).sum(), updated)
        np.testing.assert_allclose(
            next_gradient.numpy(), 2 * updated.numpy(), rtol=1e-6, atol=1e-6)

    def test1(self):
        class MyFunc(Function):
            def execute(self, x):
                return x+1

            def grad(self, grad):
                return grad-2
        a = jt.ones(1)
        func = MyFunc()
        b = func(a)
        da = jt.grad(b, a)
        assert da.data == -1

    def test_apply(self):
        class MyFunc(Function):
            def execute(self, x):
                return x+1

            def grad(self, grad):
                return grad-2
        a = jt.ones(1)
        func = MyFunc.apply
        b = func(a)
        da = jt.grad(b, a)
        assert da.data == -1

    def test2(self):
        class MyFunc(Function):
            def execute(self, x):
                self.x = x
                return x+1

            def grad(self, grad):
                return (grad-2) * self.x
        a = jt.ones(1) * 10
        func = MyFunc()
        b = func(a)
        da = jt.grad(b, a)
        assert da.data == -10

    def test_grad_not_match_error(self):
        class MyFunc(Function):
            def execute(self, x, y):
                self.x = x
                self.y = y
                return x*y

            def grad(self, grad):
                return (grad-2) * self.x
        a = jt.array(3.0)
        b = jt.array(4.0)
        func = MyFunc()
        c = func(a, b)
        expect_error(lambda: jt.grad(c, [a, b]))

    def test_multi_grads(self):
        class MyFunc(Function):
            def execute(self, x, y):
                self.x = x
                self.y = y
                return x*y

            def grad(self, grad):
                return (grad-2) * self.y, (grad-2) * self.x
        a = jt.array(3.0)
        b = jt.array(4.0)
        func = MyFunc()
        c = func(a, b)
        da, db = jt.grad(c, [a, b])
        assert da.data == -4
        assert db.data == -3

    def test_multi_grads_none(self):
        class MyFunc(Function):
            def execute(self, x, y):
                self.x = x
                self.y = y
                return x*y

            def grad(self, grad):
                return (grad-2) * self.y, None
        a = jt.array(3.0)
        b = jt.array(4.0)
        func = MyFunc()
        c = func(a, b)
        da, db = jt.grad(c, [a, b])
        assert da.data == -4
        assert db.data == 0

    def test_multi_grads_multi_out(self):
        class MyFunc(Function):
            def execute(self, x, y):
                self.x = x
                self.y = y
                return x*y, x/y

            def grad(self, grad0, grad1):
                return grad0 * self.y, grad1 * self.x
        a = jt.array(3.0)
        b = jt.array(4.0)
        func = MyFunc()
        c,d = func(a, b)
        da, db = jt.grad(c+d*3, [a, b])
        assert da.data == 4
        assert db.data == 9

    def test_multi_grads_multi_out_stop_grad_0(self):
        class MyFunc(Function):
            def execute(self, x, y):
                self.x = x
                self.y = y
                return x*y, x/y

            def grad(self, grad0, grad1):
                return grad0 * self.y, grad1 * self.x
        a = jt.array(3.0)
        b = jt.array(4.0)
        b.stop_grad()
        func = MyFunc()
        c,d = func(a, b)
        da, db = jt.grad(c+d*3, [a, b])
        assert da.data == 4
        assert db.data == 0

    def test_multi_grads_multi_out_stop_grad_1(self):
        class MyFunc(Function):
            def execute(self, x, y):
                self.x = x
                self.y = y
                return x*y, x/y

            def grad(self, grad0, grad1):
                assert grad1 is None
                return grad0 * self.y, None
        a = jt.array(3.0)
        b = jt.array(4.0)
        func = MyFunc()
        c,d = func(a, b)
        d.stop_grad()
        da, db = jt.grad(c+d*3, [a, b])
        assert da.data == 4
        assert db.data == 0

    def test_multi_grads_multi_out2(self):
        class MyFunc(Function):
            def execute(self, x, y):
                self.x = x
                self.y = y
                return x*y, x/y

            def grad(self, grad0, grad1):
                res = (grad0 * self.y, grad1 * self.x)
                print(res)
                return res
        a = jt.array(3.0)
        b = jt.array(4.0)
        func = MyFunc()
        c,d = func(a, b)
        da, db = jt.grad(c+d*3, [a, b])
        assert da.data == 4, da.data
        assert db.data == 9

    def test_multi_grads_multi_out3(self):
        class MyFunc(Function):
            def execute(self, x, y):
                self.x = x
                self.y = y
                return x*y, x/y

            def grad(self, grad0, grad1):
                res = (grad0 * self.y, grad1 * self.x)
                print(res)
                return res
        a = jt.array(3.0)
        b = jt.array(4.0)
        c,d = MyFunc()(a, b)
        da, db = jt.grad(c+d*3, [a, b])
        assert da.data == 4, da.data
        assert db.data == 9

    def test_multi_grads_multi_out4(self):
        class MyFunc(Function):
            def execute(self, x, z, y):
                self.x = x
                self.y = y
                return x*y, "test", x/y

            def grad(self, grad0, _, grad1):
                assert _ is None
                res = (grad0 * self.y, None, grad1 * self.x)
                print(res)
                return res
        a = jt.array(3.0)
        b = jt.array(4.0)
        c,_,d = MyFunc()(a, "a", b)
        da, db = jt.grad(c+d*3, [a, b])
        assert da.data == 4, da.data
        assert db.data == 9


    def test_multi_grads_multi_out5(self):
        class MyFunc(Function):
            def execute(self, x, z, y):
                self.x = x.name("x")
                self.y = y.name("y")
                return x*y, "test", x/y

            def grad(self, grad0, _, grad1):
                assert _ is None
                res = (grad0 * self.y, 1, grad1 * self.x)
                print(res)
                return res
        a = jt.array(3.0).name('a')
        b = jt.array(4.0).name('b')
        c,_,d = MyFunc()(a, "a", b)
        c.name('c'), d.name('d')
        expect_error(lambda : jt.grad(c+d*3, [a, b]))

    def test_zmem_leak(self):
        def test():
            self.test_multi_grads_multi_out5()
        test()
        jt.clean()
        self.assertEqual(jt.liveness_info()["lived_vars"], 0)

    def test_zmem_leak2(self):
        def test():
            class MyFunc(Function):
                def execute(self, x, z, y):
                    self.x = x.name("x")
                    self.y = y.name("y")
                    return x*y, "test", x/y

                def grad(self, grad0, _, grad1):
                    assert _ is None
                    res = (grad0 * self.y, None, grad1 * self.x)
                    return res
            a = jt.array(3.0).name('a')
            b = jt.array(4.0).name('b')
            c,_,d = MyFunc()(a, "a", b)
            c.name('c'), d.name('d')
            g = jt.grad(c+d*3, [a, b])
        test()
        jt.clean()
        jt.dump_all_graphs()
        self.assertEqual(jt.liveness_info()["lived_vars"], 0)

    @pytest.mark.slow
    def test_zmem_leak3(self):
        def test():
            class MyFunc(Function):
                def execute(self, x, z, y):
                    self.x = x
                    self.y = y
                    return x*y, "test", x/y

                def grad(self, grad0, _, grad1):
                    assert _ is None
                    res = (grad0 * self.y, None, grad1 * self.x)
                    return res
            a = jt.array(3.0)
            b = jt.array(4.0)
            c,_,d = MyFunc()(a, "a", b)
            g = jt.grad(c+d*3, [a, b])
            jt.sync(g)
        assert_rss_growth_bounded(
            test, iterations=512, max_growth_bytes=4 << 20, cleanup=jt.clean)
        self.assertEqual(jt.liveness_info()["lived_vars"], 0)


class TestFunctionWithEagerExecution(TestFunction):
    @classmethod
    def setUpClass(self):
        jt.flags.lazy_execution = 0
    @classmethod
    def tearDownClass(self):
        jt.flags.lazy_execution = 1

class TestFunctionCallIsIndependent(unittest.TestCase):
    """One Function instance called twice used to corrupt the first backward.

    ``execute`` saved its intermediates on ``self`` (as every example and 50+
    subclasses in this tree do), and so did the framework's own input/output
    masks. A second call overwrote both, so the FIRST call's backward ran
    against the SECOND call's tensors and returned a wrong gradient with no
    warning at all.

    ``MyFunc.apply(...)`` was accidentally safe (it builds a fresh instance per
    call) and the class docstring only shows that spelling -- but
    ``f = MyFunc(); f(x); f(y)`` is just as natural a thing to write.
    """

    class Mul(Function):
        def execute(self, a, b):
            self.a, self.b = a, b
            return a * b

        def grad(self, g):
            return g * self.b, g * self.a

    def _v(self, x):
        v = jt.array(np.array([x], dtype="float32"))
        v.start_grad()
        return v

    def test_one_instance_called_twice_keeps_both_backwards(self):
        a = self._v(1.0)
        b = self._v(2.0)
        c = self._v(10.0)
        f = self.Mul()
        o1 = f(a, b)
        o2 = f(a, c)
        # d(a*b)/da == b == 2, NOT c == 10
        np.testing.assert_allclose(
            jt.grad(o1, [a])[0].numpy(), [2.0], rtol=1e-6,
            err_msg="the second call overwrote the first call's saved state")
        np.testing.assert_allclose(
            jt.grad(o2, [a])[0].numpy(), [10.0], rtol=1e-6)

    def test_interleaved_calls_in_a_loop(self):
        f = self.Mul()
        a = self._v(1.0)
        outs, factors = [], []
        for k in range(1, 6):
            bk = self._v(float(k))
            outs.append(f(a, bk))
            factors.append(float(k))
        for out, k in zip(outs, factors):
            np.testing.assert_allclose(
                jt.grad(out, [a])[0].numpy(), [k], rtol=1e-6,
                err_msg="call %g's backward used another call's state" % k)

    def test_the_instance_is_not_mutated_by_a_call(self):
        # the call's scratch state must not leak back onto the shared instance
        f = self.Mul()
        a, b = self._v(1.0), self._v(2.0)
        f(a, b)
        for attr in ("a", "b", "input_mask", "output_mask"):
            self.assertNotIn(
                attr, f.__dict__,
                "%r leaked from the call onto the shared instance" % attr)

    def test_apply_still_works_and_agrees(self):
        a, b = self._v(1.0), self._v(3.0)
        o = self.Mul.apply(a, b)
        np.testing.assert_allclose(jt.grad(o, [a])[0].numpy(), [3.0], rtol=1e-6)

    def test_init_configuration_is_visible_to_execute(self):
        # a context is a copy of the instance, so __init__'s config survives
        class Scale(Function):
            def __init__(self, k):
                self.k = k

            def execute(self, x):
                self.x = x
                return x * self.k

            def grad(self, g):
                return g * self.k

        f = Scale(4.0)
        x = self._v(2.0)
        o = f(x)
        np.testing.assert_allclose(o.numpy(), [8.0], rtol=1e-6)
        np.testing.assert_allclose(jt.grad(o, [x])[0].numpy(), [4.0], rtol=1e-6)
        self.assertEqual(f.k, 4.0)

    def test_non_var_keyword_arguments_now_work(self):
        # __call__ used to take (*args) only, so apply(**kw) raised TypeError
        class AddK(Function):
            def execute(self, x, k=1):
                self.k = k
                return x + k

            def grad(self, g):
                return g

        x = self._v(5.0)
        np.testing.assert_allclose(
            AddK.apply(x, k=3).numpy(), [8.0], rtol=1e-6)
        np.testing.assert_allclose(
            AddK()(x, k=3).numpy(), [8.0], rtol=1e-6)

    def test_var_keyword_argument_is_refused_loudly(self):
        # only positional args are taped, so a Var by keyword would silently
        # come back with no gradient -- refuse it instead
        a, b = self._v(1.0), self._v(2.0)
        with self.assertRaises(TypeError) as cm:
            self.Mul.apply(a, b=b)
        self.assertIn("positionally", str(cm.exception))

    def test_no_grad_path_also_uses_a_context(self):
        f = self.Mul()
        a, b = self._v(1.0), self._v(2.0)
        with jt.no_grad():
            out = f(a, b)
        np.testing.assert_allclose(out.numpy(), [2.0], rtol=1e-6)
        self.assertNotIn("a", f.__dict__)


if __name__ == "__main__":
    unittest.main()
