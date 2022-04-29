# ***************************************************************
# Copyright (c) 2022 Jittor.
# All Rights Reserved. 
# Maintainers:
#     Dun Liang <randonlang@gmail.com>.
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor.test.test_grad import ngrad

class TestSlice(unittest.TestCase):
    def test_slice_bool(self):
        a = jt.zeros(10, "bool")
        a[1] = True
        a[2] = 1
        assert a.dtype == "bool"
        a.sync()
        assert np.equal(a.data, np.array([0,1,1,0,0,0,0,0,0,0])).all()

    def test_var_slices(self):
        def check(slices, msg):
            with jt.log_capture_scope() as logs:
                jt.core._print_var_slice(slices)
            s = logs[0]['msg']
            assert s == msg, s
        check((1), "[1,]")
        check(([[0],[1]],slice(None),[1,2],1), "[int32[2,1,],::,int32[2,],1,]")
        check((slice(None),slice(None),slice(None),slice(None)), "[::,::,::,::,]")
        check(([0,1],[0,1],[0,1],[0,1]), "[int32[2,],int32[2,],int32[2,],int32[2,],]")
        check(([0,1],-2,slice(None),[0,1]), "[int32[2,],-2,::,int32[2,],]")
        check(([0,1],slice(1,2,2),[1,2],1), "[int32[2,],1:2:2,int32[2,],1,]")
        check(([0,1],slice(None),[1,2],1), "[int32[2,],::,int32[2,],1,]")
        check((slice(1,None,2),slice(-1,None,2),[1,2],-4), "[1::2,-1::2,int32[2,],-4,]")
        check(0, "[0,]")
        check(10, "[10,]")
        check(-10, "[-10,]")
        check(1, "[1,]")
        check((1,slice(None),2), "[1,::,2,]")
        check((-2,slice(None),2,slice(1,9,2)), "[-2,::,2,1:9:2,]")
        check((None,1,None,2,None), "[-,1,-,2,-,]")
        check((...,1,...,2,...), "[...,1,...,2,...,]")

    @unittest.skipIf(not jt.has_cuda, "No cuda")
    @jt.flag_scope(use_cuda=1)
    def test_getitem(self):
        def check(shape, slices, i_to_vs="", i_to_o="", o_shape=""):
            # print(slices)
            x = jt.random(shape)

            with jt.log_capture_scope(log_vprefix="getitem=999") as logs:
                a = x.getitem(slices)
                a.sync()
            b = x.data[slices]
            bshape = b.shape if len(b.shape) else (1,)
            assert a.shape == bshape, (a.shape, bshape)
            s = logs[-1]['msg']
            assert "i_to_vs: "+i_to_vs in s
            assert "i_to_o: "+i_to_o in s
            assert "o_shape: "+o_shape in s
            aa = a.numpy()
            assert (aa==b).all(), (aa, b)

            y = x.numpy()
            v = jt.random(a.shape)
            z = x.setitem(slices, v)
            y[slices] = v.data
            assert (z.data==y).all(), (z.data, y, v.data, x.data)

            # test_setitem broadcast
            adim = len(a.shape)
            for mask in range(1<<adim):
                new_shape = list(a.shape)
                for i in range(adim):
                    if (mask>>i)&1:
                        new_shape[i] = 1
                y = x.numpy()
                v = jt.random(new_shape)
                z = x.setitem(slices, v)
                y[slices] = v.data
                assert (z.data==y).all(), (z.data, y, v.data, x.data)
                

        # TODO: when slice same row/col many times and assign value, numpy will retain the last value but we assign their sum. eg: check([3,3,3,3], ([[0,1,1]],slice(None),[[1],[2],[0]],1))
        check([3], (1), "[0,]", "[-1,]", "[]")
        check([3,3,3,3], ([[0],[1]],slice(None),[1,2],1), "[0,-1,2,3,]", "[-1,2,-1,-1,]", "[2,2,3,]")
        check([3,3,3,3], (slice(None),slice(None),slice(None),slice(None)), "[-1,-2,-2,-2,]", "[0,0,0,0,]", "[81,]")
        check([3,3,3,3], ([0,1],[0,1],[0,1],[0,1]), "[0,1,2,3,]", "[-1,-1,-1,-1,]", "[2,]")
        check([3,3,3,3], ([0,1],-2,slice(None),[0,1]), "[0,1,-1,3,]", "[-1,-1,1,-1,]", "[2,3,]")
        check([3,3,3,3], ([0,1],slice(1,2,2),[1,2],1), "[0,1,2,3,]", "[-1,1,-1,-1,]", "[2,1,]")
        check([3,3,3,3], ([0,1],slice(None),[1,2],1), "[0,-1,2,3,]", "[-1,1,-1,-1,]", "[2,3,]")
        check([3,3,3,3], (slice(1,10,1),...,slice(2,None,-1)), "[0,-1,-2,2,]", "[0,1,1,2,]", "[2,9,3,]")
        check([10,10,10,10], (slice(1,None,2),slice(-1,None,2),[1,2],-4), "[0,1,2,3,]", "[0,1,-1,-1,]", "")
        check([20], 0, "[0,]", "[-1,]", "[]")
        check([20], 10, "[0,]", "[-1,]", "[]")
        check([20], -10, "[0,]", "[-1,]", "[]")
        check([10,10,10,10], 1, "[0,-1,-2,-2,]", "[-1,0,0,0,]", "[1000,]")
        check([10,10,10,10], (1,slice(None),2), "[0,-1,2,-1,]", "[-1,0,-1,1,]", "")
        check([10,10,10,10], (-2,slice(None),2,slice(1,9,2)), "[0,-1,2,3,]", "[-1,0,-1,1,]")

    def test_getitem_grad(self):
        shape = (10,)
        slices = slice(2,4)

        a = jt.random(shape)
        b = a.getitem(slices)
        mask = jt.random(b.shape)
        loss = b*mask
        da = jt.grad(loss, a)

        _, np_grad = ngrad(lambda vars: (vars[0][slices]*mask.data).sum(), [a.numpy()], 1e-3)
        assert np.allclose(da.numpy(), np_grad, atol = 1e-3), (da.numpy(), np_grad)

        shape = (10,)
        slices = slice(2,4)

        a = jt.random(shape)
        b = a.getitem(slices)
        b = jt.random(b.shape)
        c = a.setitem(slices, b)
        mask = jt.random(c.shape)
        loss = c*mask
        da,db = jt.grad(loss, [a,b])

        def numpy_grad(vars):
            a, b = vars
            a = a.copy()
            a[slices] = b
            return (a*mask.data).sum()

        _, (nda, ndb) = ngrad(numpy_grad, [a.numpy(), b.numpy()], 1e-3)
        assert np.allclose(da.numpy(), nda, atol = 1e-3)
        assert np.allclose(db.numpy(), ndb, atol = 1e-3)

    def test_vary_shape_setitem(self):
        a = jt.array([1,2,3,4,5])
        b = jt.array([1,2,3,4,5])
        c = tuple(jt.where(b>3))
        a[c] = 0
        assert (a.data == [1,2,3,0,0]).all()

    def test_numpy_scalar_slice(self):
        a = jt.random((2,2))
        b = np.array([1])[0]
        assert a[b].shape == [2]



if __name__ == "__main__":
    unittest.main()
