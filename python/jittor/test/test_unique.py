
# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
#     Xiangli Li <1905692338@qq.com>
#     Jiapeng Zhang <zhangjp20@mails.tsinghua.edu.cn>
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

from cgi import test
import unittest
import jittor as jt
import numpy as np

skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
except:
    torch = None
    skip_this_test = True

def test_unique_with_torch(input, dim=None):
    jt0, jt1, jt2 = jt.unique(jt.array(input), True, True, dim)
    torch0, torch1, torch2 = torch.unique(torch.tensor(input), True, True, True, dim)
    assert np.allclose(jt0, torch0) and np.allclose(jt1, torch1) and np.allclose(jt2, torch2)


@unittest.skipIf(skip_this_test, "No Torch found")
class TestSparse(unittest.TestCase):

    def test_unique(self):
        test_unique_with_torch(np.array([1, 3, 2, 3, 3, 3], dtype=np.int32))
        test_unique_with_torch(np.array([[1, 3], [2, 3], [1, 2]], dtype=np.int64))

    def test_unique_dim(self):
        test_unique_with_torch(np.array([[1, 3], [2, 3], [1, 3], [2, 3]]), 0)
        test_unique_with_torch(np.array([[1, 3], [2, 3], [1, 3], [2, 3]]), 1)


    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_unique_cuda(self):
        self.test_unique()

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_unique_dim_cuda(self):
        self.test_unique_dim()
    
        
if __name__ == "__main__":
    unittest.main()