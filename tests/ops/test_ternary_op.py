# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from _helpers.assertions import expect_error
from _helpers.devices import cuda_test_case
from _helpers.numerical_grad import ngrad

class TestTernaryOp(unittest.TestCase):
    def test_shape_mismatch_is_a_catchable_user_error(self):
        with self.assertRaisesRegex(RuntimeError, "Shape not match"):
            jt.ternary(
                jt.ones((2, 3)), jt.ones((3, 2)), jt.zeros((2, 3)))

    def test_with_np(self):
        np.random.seed(0)
        a = np.random.rand(5,10).astype("float32")
        b = np.random.rand(5,10).astype("float32")
        ja = jt.array(a)
        jb = jt.array(b)
        jc = jt.ternary(ja>jb, ja, jb)
        assert (jc.data==np.maximum(a,b)).all(), f"\n{jc.data}\n{np.maximum(a,b)}\n{a}\n{b}"
        jda, jdb = jt.grad(jc, [ja, jb])
        assert (jda.data==(a>b)*1).all()
        assert (jdb.data==1-(a>b)).all()

    def test_where(self):
        np.random.seed(0)
        a = np.random.rand(5,10).astype("float32")
        b = np.random.rand(5,10).astype("float32")
        ja = jt.array(a)
        jb = jt.array(b)
        jc = jt.where(ja>jb, ja, jb)
        assert (jc.data==np.maximum(a,b)).all(), f"\n{jc.data}\n{np.maximum(a,b)}\n{a}\n{b}"
        jda, jdb = jt.grad(jc, [ja, jb])
        assert (jda.data==(a>b)*1).all()
        assert (jdb.data==1-(a>b)).all()

    def test_min(self):
        np.random.seed(1)
        a = np.random.rand(5,10).astype("float32")
        b = np.random.rand(5,10).astype("float32")
        ja = jt.array(a)
        jb = jt.array(b)
        jc = jt.minimum(ja,jb)
        assert (jc.data==np.minimum(a,b)).all(), f"\n{jc.data}\n{np.minimum(a,b)}\n{a}\n{b}"
        jda, jdb = jt.grad(jc, [ja, jb])
        assert (jda.data==(a<b)*1).all()
        assert (jdb.data==1-(a<b)).all()

class TestTernaryOpCuda(TestTernaryOp, cuda_test_case(2)):
    pass

if __name__ == "__main__":
    unittest.main()
