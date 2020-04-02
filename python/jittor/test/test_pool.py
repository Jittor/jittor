# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor import nn as jnn
import numpy as np

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    from torch import nn as tnn
except:
    torch = None
    
skip_this_test = False

@unittest.skipIf(skip_this_test, "skip_this_test")
class TestPool(unittest.TestCase):
    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1, use_stat_allocator=1)
    def test_pool(self):
        test_img = np.random.random((16,3,224,224)).astype('float32')
        # Define pytorch & jittor input image
        pytorch_test_img = torch.Tensor(test_img).cuda()
        jittor_test_img = jt.array(test_img)
        pytorch_pool = tnn.AvgPool2d(2,2)
        jittor_pool = jnn.Pool(2,2,op="mean")


if __name__ == "__main__":
    unittest.main()
