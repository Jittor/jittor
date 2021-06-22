# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

skip_this_test = False
try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    from torch.autograd import Variable
except:
    torch = None
    skip_this_test = True


@unittest.skipIf(skip_this_test, "No Torch found")
class TestCumprod(unittest.TestCase):
    def test_cumprod_cpu(self):
        for i in range(1,6):
            for j in range(i):
                x = np.random.rand(*((10,)*i))
                x_jt = jt.array(x)
                y_jt = jt.cumprod(x_jt, j).sqr()
                g_jt = jt.grad(y_jt.sum(), x_jt)
                x_tc = Variable(torch.from_numpy(x), requires_grad=True)
                y_tc = torch.cumprod(x_tc, j)**2
                y_tc.sum().backward()
                g_tc = x_tc.grad
                assert np.allclose(y_jt.numpy(), y_tc.data)
                assert np.allclose(g_jt.numpy(), g_tc.data)

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_cumprod_gpu(self):
        self.test_cumprod_cpu()

if __name__ == "__main__":
    unittest.main()