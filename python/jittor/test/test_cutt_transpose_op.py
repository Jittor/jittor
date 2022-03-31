# ***************************************************************
# Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error
from .test_grad import ngrad
from itertools import permutations
from jittor import compile_extern
from .test_log import find_log_with_re
if jt.has_cuda:
    from jittor.compile_extern import cutt_ops
else:
    cutt_ops = None

def gen_data(shape):
    num = np.multiply.reduce(shape)
    a = np.arange(0, num)
    return a.reshape(shape)

class TestCuttTransposeOp(unittest.TestCase):
    @unittest.skipIf(cutt_ops==None, "Not use cutt, Skip")
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
        
    @unittest.skipIf(cutt_ops==None, "Not use cutt, Skip")
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
        
    @unittest.skipIf(cutt_ops==None, "Not use cutt, Skip")
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

if __name__ == "__main__":
    unittest.main()