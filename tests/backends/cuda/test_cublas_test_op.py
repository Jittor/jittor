
from _helpers import capability as _test_capability
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
from jittor import compile_extern

@_test_capability.library_required("cublas", backend=jt)
class TestCublasTestOp(unittest.TestCase):
    def test(self):
        from jittor.compile_extern import cublas_ops
        assert cublas_ops.cublas_test(2).data==123
        assert cublas_ops.cublas_test(5).data==123
        assert cublas_ops.cublas_test(10).data==123
        assert cublas_ops.cublas_test(20).data==123

@_test_capability.library_required("cudnn", backend=jt)
class TestCudnnTestOp(unittest.TestCase):
    def test(self):
        from jittor.compile_extern import cudnn_ops
        assert cudnn_ops.cudnn_test("").data == 123
        assert cudnn_ops.cudnn_test("-c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1").data == 123
        
@_test_capability.library_required("cub", backend=jt)
class TestCubTestOp(unittest.TestCase):
    @jt.flag_scope(use_cuda=1)
    def test(self):
        from jittor.compile_extern import cub_ops
        assert cub_ops.cub_test("xx").data == 123
        assert cub_ops.cub_test("xx --n=100000").data == 123
        
if __name__ == "__main__":
    unittest.main()
