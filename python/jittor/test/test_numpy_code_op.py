# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

class TestCodeOp(unittest.TestCase):
    def forward_code(np, data):
        a,b = data["inputs"]
        c,d = data["outputs"]
        np.add(a,b,out=c)
        np.substract(a,b,out=d)

    def backward_code1(np, data):
        dout = data["dout"]
        da, db = data["outputs"]
        np.copyto(dout, da)
        np.copyto(dout, db)

    def backward_code2(np, data):
        dout = data["dout"]
        da, db = data["outputs"]
        np.copyto(dout, da)
        np.negtive(dout, db)

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

        print("a:",a)
        print("b:",b)
        print("a+b:",c)
        print("a-b:",d)

if __name__ == "__main__":
    unittest.main()