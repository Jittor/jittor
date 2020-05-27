# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
from jittor import compile_extern
if jt.has_cuda:
    from jittor.compile_extern import cublas_ops, cudnn_ops, cub_ops
else:
    cublas_ops = cudnn_ops = cub_ops = None

@unittest.skipIf(cublas_ops==None, "Not use cublas, Skip")
class TestCublasTestOp(unittest.TestCase):
    def test(self):
        assert cublas_ops.cublas_test(2).data==123
        assert cublas_ops.cublas_test(5).data==123
        assert cublas_ops.cublas_test(10).data==123
        assert cublas_ops.cublas_test(20).data==123

@unittest.skipIf(cudnn_ops==None, "Not use cudnn, Skip")
class TestCudnnTestOp(unittest.TestCase):
    def test(self):
        assert cudnn_ops.cudnn_test("").data == 123
        assert cudnn_ops.cudnn_test("-c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1").data == 123
        
@unittest.skipIf(cub_ops==None, "Not use cub, Skip")
class TestCubTestOp(unittest.TestCase):
    @jt.flag_scope(use_cuda=1)
    def test(self):
        assert cub_ops.cub_test("xx").data == 123
        assert cub_ops.cub_test("xx --n=100000").data == 123
        
if __name__ == "__main__":
    unittest.main()
