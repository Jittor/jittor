# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import unittest

@unittest.skipIf(jt.compile_extern.nccl_ops is None, "no nccl found")
class TestNccl(unittest.TestCase):
    @jt.flag_scope(use_cuda=1)
    def test_nccl(self):
        assert jt.compile_extern.nccl_ops.nccl_test("").data == 123

if __name__ == "__main__":
    unittest.main()
