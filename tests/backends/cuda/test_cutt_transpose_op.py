# ***************************************************************
# Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from _helpers.assertions import expect_error
from _helpers.numerical_grad import ngrad
from itertools import permutations
from jittor import compile_extern
from _helpers.cutt import require_cutt_ops
from _helpers.logs import find_log_with_re

def gen_data(shape):
    num = np.multiply.reduce(shape)
    a = np.arange(0, num)
    return a.reshape(shape)

class TestCuttTransposeOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cutt_ops = require_cutt_ops()

    @jt.flag_scope(use_cuda=1)
    def test_axes_length_is_a_catchable_user_error(self):
        x = jt.array(np.zeros((2, 3), dtype="float32"))
        # `[0]` counts up from 0, so the constructor's identity test used to
        # accept it and forward x unchanged: the rank check in infer_shape was
        # unreachable and a wrong axes argument silently did nothing.
        for axes in ([0], [0, 1, 2]):
            expect_error(
                lambda: self.cutt_ops.cutt_transpose(x, axes).sync(),
                exc_type=RuntimeError,
                match="axes.size",
            )

    @jt.flag_scope(use_cuda=1)
    def test_duplicate_axes_are_a_catchable_user_error(self):
        x = jt.array(np.zeros((2, 3), dtype="float32"))
        expect_error(
            lambda: self.cutt_ops.cutt_transpose(x, [0, 0]),
            exc_type=RuntimeError,
            match="Invalid axes",
        )

    @jt.flag_scope(use_cuda=1)
    def test_the_runtime_still_computes_after_a_rejected_axes(self):
        x = jt.array(np.zeros((2, 3), dtype="float32"))
        expect_error(
            lambda: self.cutt_ops.cutt_transpose(x, [0]).sync(),
            exc_type=RuntimeError,
            match="axes.size",
        )
        self.assertEqual(float((jt.ones((4, 4)) * 2).sum().item()), 32.0)

    @jt.flag_scope(use_cuda=1)
    def test_identity_axes_forward_the_input(self):
        a = gen_data([2, 3])
        x = jt.array(a).float()
        np.testing.assert_allclose(
            self.cutt_ops.cutt_transpose(x, [0, 1]).data, a)

    @jt.flag_scope(use_cuda=1)
    def test_with_np(self):
        def check(a):
            perms = list(permutations(range(a.ndim))) + [None]
            for perm in perms:
                with jt.log_capture_scope(
                    log_silent=1,
                    log_v=0, log_vprefix="cutt=100"
                ) as raw_log:
                    if perm:
                        x = np.transpose(a, perm)
                        y = jt.transpose(a, perm).data
                    else:
                        x = np.transpose(a)
                        y = jt.transpose(a).data
                    self.assertEqual(x.shape, y.shape)
                logs = find_log_with_re(raw_log, "(Run cutt_transpose with key.*)")
                if perm is None:
                    continue
                last = -1
                in_order = True
                for i in range(len(perm)):
                    if a.shape[perm[i]] == 1:
                        continue
                    if last != -1 and last > perm[i]:
                        in_order = False
                        break
                    last = perm[i]
                # if not in_order:
                #     assert len(logs)==1
                assert (x==y).all(), f"\n{x}\n{y}\n{perm}\n{a.shape}"
                
        ia = [gen_data([5, 7]), gen_data([2,2,2]), gen_data([2,3,4,5]), gen_data([5,3]), gen_data([3,1,5,3,1])]
        for a in ia: check(a)
        
    @jt.flag_scope(use_cuda=1)
    def test_grad(self):
        def check(a):
            perms = list(permutations(range(a.ndim))) + [None]
            for perm in perms:
                x = jt.array(a).float()
                if perm:
                    y = jt.transpose(x, perm)
                else:
                    y = jt.transpose(x)
                dx = jt.grad(y*y, x).data
                self.assertEqual(dx.shape, a.shape)
                assert (dx==a*2).all(), f"\n{dx}\n{a}\n{perm}"
        ia = [gen_data([2,2,2]), gen_data([2,3,4,5]), gen_data([5,3]), gen_data([3,1,5,3,1])]
        for a in ia: check(a)
        
    # Was a second `def test_matmul_grad`, which shadowed the numerical-gradient
    # case below it and kept it from ever running -- the same "entry that is
    # never executed is not an entry" as the permanent cuTT skip.
    @jt.flag_scope(use_cuda=1)
    def test_matmul_grad(self):
        np.random.seed(0)
        for i in range(10):
            a = np.random.rand(2,3).astype("float32")
            b = np.random.rand(3,4).astype("float32")
            out, (da, db) = ngrad(lambda vars: np.matmul(vars[0],vars[1]).sum(), [a,b], 1e-1)
            ja = jt.array(a)
            jb = jt.array(b)
            jc = ja.matmul(jb)
            jda, jdb = jt.grad(jc, [ja,jb])
            assert ((da-jda.data)<1e-5).all(), (da, jda.data, da-jda.data)
            assert ((db-jdb.data)<1e-5).all(), (db-jdb.data)

    @jt.flag_scope(use_cuda=1)
    def test_empty_matmul_transpose(self):
        a = jt.zeros((0, 10))
        b = a.transpose(1, 0)
        c = b.data
        assert c.shape[0] == 10
        assert c.shape[1] == 0

if __name__ == "__main__":
    unittest.main()
