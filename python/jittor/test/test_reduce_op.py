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

def gen_data(shape):
    num = np.multiply.reduce(shape)
    a = np.arange(0, num)
    return a.reshape(shape).astype("int32")

    
class TestReduceOp(unittest.TestCase):
    def setUp(self):
        self.keepdims = False
        
    def test1(self):
        def check(a, op, dims):
            if ("logical" in op) and jt.flags.use_cuda:
                # TODO: atomic bool operation for cuda not 
                # supported yet
                return
            np_dims = jt_dims = dims
            if dims == ():
                np_dims = tuple(range(len(a.shape)))
            x = eval(f"np.{op}.reduce(a, {np_dims}, keepdims={self.keepdims})")
            y = eval(f"jt.reduce_{op}(a, {jt_dims}, keepdims={self.keepdims}).data")
            if len(x.shape) == 0: x = np.array([x]).astype(a.dtype)
            x = x.astype(a.dtype)
            y = y.astype(a.dtype)
            assert x.dtype == y.dtype and x.shape == y.shape and (x==y).all(), \
                f"\n{a.shape}\n{op}\n{dims}\n{x}\n{y}\n{x.dtype}\n{y.dtype}\n{a.dtype}"
                
        ia = [gen_data([2,3,4,5]), gen_data([5,3])]
        idims = [(), (0,), (1,), (2,), (3,), (0, 2), (1,3), (1,2,3), 2, 3]
        
        iop = [ op[7:] for op in dir(jt) if op.startswith("reduce_")]
        assert len(iop) >= 10, iop
        for a in ia:
            check(a, iop[0], idims[0])
        for op in iop:
            check(ia[0], op, idims[0])
        for dims in idims:
            check(ia[0], iop[0], dims)
        expect_error(lambda: jt.reduce_add([1,2,3], 2))


class TestReduceOp2(TestReduceOp):
    def setUp(self):
        self.keepdims = True


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestReduceOpCuda(TestReduceOp):
    def setUp(self):
        jt.flags.use_cuda = 2
        self.keepdims = False
    def tearDown(self):
        jt.flags.use_cuda = 0

@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestReduceOpCuda2(TestReduceOp):
    def setUp(self):
        jt.flags.use_cuda = 2
        self.keepdims = True
    def tearDown(self):
        jt.flags.use_cuda = 0


class TestReduceOpMisc(unittest.TestCase):
    def test_negtive_dim(self):
        a = jt.array([[1,2],[3,4]])
        assert (a.sum(-1).data == [3,7]).all()
        assert (a.sum(-2).data == [4,6]).all()

if __name__ == "__main__":
    unittest.main()
