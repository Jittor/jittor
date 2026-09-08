# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import unittest
from _helpers import capability as _test_capability

@_test_capability.library_required("nccl", backend=jt)
class TestNccl(unittest.TestCase):
    @jt.flag_scope(use_cuda=1)
    def test_nccl(self):
        assert jt.compile_extern.nccl_ops.nccl_test("").data == 123

if __name__ == "__main__":
    unittest.main()
