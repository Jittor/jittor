# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error
ops = jt.ops

@jt.var_scope('linear')
def linear(x, n):
    w = jt.make_var([x.shape[-1], n], init=ops.random)
    return jt.matmul(x, w)

@jt.var_scope('model', unique=True)
def model(x):
    x = linear(x, 10)
    # x = relu(x)
    x = linear(x, 10)
    # x = relu(x)
    x = linear(x, 1)
    return x

class TestScope(unittest.TestCase):
    def test_name(self):
        jt.clean()
        @jt.var_scope('model', unique=True)
        def model():
            with jt.var_scope('a'):
                assert jt.current_scope.full_name == "model/a_0/"
            with jt.var_scope('b'):
                with jt.var_scope('b'):
                    assert jt.current_scope.full_name == "model/b_0/b_0/"
            with jt.var_scope('c'):
                assert jt.current_scope.full_name == "model/c_0/"
        model()
        model()
        model()
        jt.clean()
        
    def test_var(self):
        jt.clean()
        for i in range(2):
            x = jt.array([[1]])
            y = model(x)
            params = jt.find_vars()
            assert len(params) == 3
            names = [ p.name() for p in params ]
            assert names == [
                "model/linear_0/var_0",
                "model/linear_1/var_0",
                "model/linear_2/var_0",
            ], str(names)
            jt.find_var("model/linear_0/var_0")
            expect_error(lambda: jt.find_var("model/linear_0"))
            expect_error(lambda: jt.find_var("model/linear"))
            assert len(jt.find_vars("model/linear_0/var_0")) == 1
            assert len(jt.find_vars("model/linear_0/")) == 1
            assert len(jt.find_vars("model/")) == 3
        jt.clean()
        
    def test_get_var_unique(self):
        jt.clean()
        x = jt.make_var([1], init=ops.random)
        y = jt.make_var([1], init=ops.random)
        z = jt.make_var([1], init=ops.random)
        assert x.name() == "var_0"
        assert y.name() == "var_1", y.name()
        assert z.name() == "var_2"
        x = jt.make_var([1], name="x", unique=True, init=ops.random)
        y = jt.make_var([1], name="y", unique=True, init=ops.random)
        z = jt.make_var([1], name="z", unique=True, init=ops.random)
        assert x.name() == "x"
        assert y.name() == "y"
        assert z.name() == "z"
        expect_error(lambda: jt.make_var([2], name="x", unique=True, init=ops.random))
        jt.clean()

    def test_record_scope(self):
        jt.clean()
        @jt.var_scope("func")
        def func(a):
            b = a+1
            jt.record_in_scope(b, "b")
            c = b*2
            return c
        a = jt.array([1,2,3])
        func(a)
        assert np.allclose(jt.find_record("func_0/output").data, (a.data+1)*2)
        assert np.allclose(jt.find_scope("func_0").records["output"].data, (a.data+1)*2)
        recs = jt.find_records()
        rec_names = [ r.name() for r in recs ]
        assert len(recs)==2 and rec_names==["func_0/b","func_0/output"]
        
    def test_get_var_init(self):
        jt.clean()
        assert (jt.make_var(init=[1,2,3]).data == [1,2,3]).all()
        assert (jt.make_var(shape=[3], init=np.zeros).data == [0,0,0]).all()
        assert (jt.make_var(init=jt.array([1,2,3]) == [1,2,3]).data).all()
        jt.clean()

if __name__ == "__main__":
    unittest.main()