# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from collections.abc import Sequence, Mapping
from .test_core import expect_error
from jittor import Function

class TestFunction(unittest.TestCase):
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
                self.x = x
                self.y = y
                return x*y, "test", x/y

            def grad(self, grad0, _, grad1):
                assert _ is None
                res = (grad0 * self.y, 1, grad1 * self.x)
                print(res)
                return res
        a = jt.array(3.0)
        b = jt.array(4.0)
        c,_,d = MyFunc()(a, "a", b)
        expect_error(lambda : jt.grad(c+d*3, [a, b]))

    def test_zz_last_test(self):
        self.assertEqual(jt.liveness_info()["lived_vars"], 0)

if __name__ == "__main__":
    unittest.main()