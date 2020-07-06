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
    def forward_code(self, np, data):
        a,b = data["inputs"]
        c,d = data["outputs"]
        np.add(a,b,out=c)
        np.subtract(a,b,out=d)
        p, r = c.__array_interface__['data']

    def backward_code1(self, np, data):
        dout = data["dout"]
        a,b,dout = data["inputs"]
        out = data["outputs"][0]
        np.copyto(out, dout)

    def backward_code2(self, np, data):
        dout = data["dout"]
        out_index = data["out_index"]
        out = data["outputs"][0]
        if out_index==0:
            np.copyto(out, dout)
        else:
            np.negative(dout, out)

    def test(self):
        a = jt.random((5,1))
        b = jt.random((5,1))

        c, d = jt.numpy_code(
            [a.shape, a.shape],
            [a.dtype, a.dtype],
            [a, b],
            self.forward_code,
            [self.backward_code1,self.backward_code2],
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