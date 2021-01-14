# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os

@unittest.skipIf(not jt.compile_extern.use_mkl, "Not use mkl, Skip")
class TestMklTestOp(unittest.TestCase):
    def test(self):
        assert jt.mkl_ops.mkl_test().data==123

if __name__ == "__main__":
    unittest.main()
