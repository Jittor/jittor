
# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import jittor.nn as jnn

skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torch.nn as tnn
    import torchvision
except:
    torch = None
    tnn = None
    torchvision = None
    skip_this_test = True

# TODO: more test
# @unittest.skipIf(skip_this_test, "No Torch found")
class TestSearchSorted(unittest.TestCase):
    def test_origin(self):
        sorted = jt.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
        values = jt.array([[3, 6, 9], [3, 6, 9]])
        ret = jt.searchsorted(sorted, values)
        assert (ret == [[1, 3, 4], [1, 2, 4]]).all(), ret

        ret = jt.searchsorted(sorted, values, right=True)
        assert (ret == [[2, 3, 5], [1, 3, 4]]).all(), ret

        sorted_1d = jt.array([1, 3, 5, 7, 9])
        ret = jt.searchsorted(sorted_1d, values)
        assert (ret == [[1, 3, 4], [1, 3, 4]]).all(), ret

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda(self):
        self.test_origin()

        

if __name__ == "__main__":
    unittest.main()