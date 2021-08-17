# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error
from .test_grad import ngrad
from itertools import permutations
from jittor.test.test_cuda import test_cuda

def gen_data(shape):
    num = np.multiply.reduce(shape)
    a = np.arange(0, num)
    return a.reshape(shape)

class TestTransposeOp(unittest.TestCase):
    def test_with_np(self):
        def check(a):
            perms = list(permutations(range(a.ndim))) + [None]
            for perm in perms:
                if perm:
                    x = np.transpose(a, perm)
                    y = jt.transpose(a, perm).data
                else:
                    x = np.transpose(a)
                    y = jt.transpose(a).data
                self.assertEqual(x.shape, y.shape)
                assert (x==y).all(), f"\n{x}\n{y}"
                
        # ia = [gen_data([2,3,4,5]), gen_data([5,3])]
        ia = [gen_data([2,2,2]), gen_data([2,3,4,5]), gen_data([5,3])]
        for a in ia: check(a)
        
    def test_grad(self):
        def check(a):
            perms = list(permutations(range(a.ndim))) + [None]
            for perm in perms:
                x = jt.array(a).float()
                if perm:
                    y = x.transpose(perm)
                else:
                    y = x.transpose()
                dx = jt.grad(y*y, x).data
                self.assertEqual(dx.shape, a.shape)
                assert (dx==a*2).all(), f"\n{dx}\n{a}\n{perm}"
        ia = [gen_data([2,2,2]), gen_data([2,3,4,5]), gen_data([5,3])]
        for a in ia: check(a)
        
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

    def test_permute(self):
        a = jt.ones([2,3,4])
        assert a.permute().shape == [4,3,2]
        assert a.permute(0,2,1).shape == [2,4,3]

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_cutt(self):
        a = jt.rand((10,2)) > 0.5
        b = a.transpose()
        assert (a.data.transpose() == b.data).all()

        a = jt.zeros((1,1))
        b = a.transpose((1,0))
        b.sync()


class TestFuseTransposeOp(unittest.TestCase):

    def test_fuse_transpose1(self):
        with jt.profile_scope() as rep:
            a = jt.rand((10,11,12))
            b = a.fuse_transpose((1,2,0))+1
            np.testing.assert_allclose(
                a.data.transpose((1,2,0))+1,
                b.data
            )
        assert len(rep) == 3

    def test_fuse_transpose2(self):
        with jt.profile_scope() as rep:
            a = jt.rand((10,11,12))
            b = (a+1).fuse_transpose((1,2,0))
            np.testing.assert_allclose(
                a.data.transpose((1,2,0))+1,
                b.data
            )
        assert len(rep) == 3

    def test_fuse_transpose3(self):
        with jt.profile_scope() as rep:
            a = jt.rand((10,11,12))
            c = jt.rand((11,12,10))
            b = a.fuse_transpose((1,2,0))+c
            np.testing.assert_allclose(
                a.data.transpose((1,2,0))+c.data,
                b.data
            )
        assert len(rep) == 3

    def test_fuse_transpose4(self):
        with jt.profile_scope() as rep:
            a = jt.rand((10,11,12))
            c = jt.rand((10,11,12))
            b = (a+c).fuse_transpose((1,2,0))
            np.testing.assert_allclose(
                (a.data+c.data).transpose((1,2,0)),
                b.data
            )
        assert len(rep) == 3

    def test_fuse_transpose5(self):
        with jt.profile_scope() as rep:
            a = jt.rand((10,11,6,7))
            c = jt.rand((10,11,6,7))
            b = (a+c).fuse_transpose((1,0,2,3))
            np.testing.assert_allclose(
                (a.data+c.data).transpose((1,0,2,3)),
                b.data
            )
        assert len(rep) == 3


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestFuseTransposeCudaOp(TestFuseTransposeOp):
    def setUp(self):
        jt.flags.use_cuda = 1
    def tearDown(self):
        jt.flags.use_cuda = 0

if __name__ == "__main__":
    unittest.main()