# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

class TestCodeOp(unittest.TestCase):
    def test(self):
        def forward_code(np, data):
            a = data["inputs"][0]
            b = data["outputs"][0]
            np.add(a,a,out=b)

        def backward_code(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout*2.0)

        a = jt.random((5,1))
        b = jt.numpy_code(
            a.shape,
            a.dtype,
            [a],
            forward_code,
            [backward_code],
        )
        assert np.allclose(b.data,(a+a).data)
        da = jt.grad(b,a)
        one=np.ones(a.shape)
        assert np.allclose(da.data,one*2.0)

    def test_multi_input(self):
        def forward_code(np, data):
            a,b = data["inputs"]
            c,d = data["outputs"]
            np.add(a,b,out=c)
            np.subtract(a,b,out=d)

        def backward_code1(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout)

        def backward_code2(np, data):
            dout = data["dout"]
            out_index = data["out_index"]
            out = data["outputs"][0]
            if out_index==0:
                np.copyto(out, dout)
            else:
                np.negative(dout, out)

        a = jt.random((5,1))
        b = jt.random((5,1))
        c, d = jt.numpy_code(
            [a.shape, a.shape],
            [a.dtype, a.dtype],
            [a, b],
            forward_code,
            [backward_code1,backward_code2],
        )
        assert np.allclose(c.data,(a+b).data)
        assert np.allclose(d.data,(a-b).data)
        dca, dcb = jt.grad(c,[a,b])
        dda, ddb = jt.grad(d,[a,b])
        one=np.ones(a.shape)
        mone=one*-1.0
        assert np.allclose(dca.data,one)
        assert np.allclose(dcb.data,one)
        assert np.allclose(dda.data,one)
        assert np.allclose(ddb.data,mone)

    @unittest.skipIf(True, "Memory leak testing is not in progress, Skip")
    def test_memory_leak(self):
        def forward_code(np, data):
            a,b = data["inputs"]
            c,d = data["outputs"]
            np.add(a,b,out=c)
            np.subtract(a,b,out=d)

        def backward_code1(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout)

        def backward_code2(np, data):
            dout = data["dout"]
            out_index = data["out_index"]
            out = data["outputs"][0]
            if out_index==0:
                np.copyto(out, dout)
            else:
                np.negative(dout, out)

        for i in range(1000000):
            a = jt.random((10000,1))
            b = jt.random((10000,1))
            c, d = jt.numpy_code(
                [a.shape, a.shape],
                [a.dtype, a.dtype],
                [a, b],
                forward_code,
                [backward_code1,backward_code2],
            )
            assert np.allclose(c.data,(a+b).data)
            assert np.allclose(d.data,(a-b).data)
            dca, dcb = jt.grad(c,[a,b])
            dda, ddb = jt.grad(d,[a,b])
            one=np.ones(a.shape)
            mone=one*-1.0
            assert np.allclose(dca.data,one)
            assert np.allclose(dcb.data,one)
            assert np.allclose(dda.data,one)
            assert np.allclose(ddb.data,mone)

if __name__ == "__main__":
    unittest.main()