# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
import numpy as np

from jittor.test.test_log import find_log_with_re
skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
except:
    skip_this_test = True

@unittest.skipIf(skip_this_test, "No Torch found")
class TestConvTranspose(unittest.TestCase):

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda(self):
        self.test()

    def test(self):
        def check(data_shape, weights_shape, stride=1, dilation=1):
            N,C,H,W = data_shape
            i,o,h,w = weights_shape
            img = np.random.rand(N,C,H,W).astype("float32")
            weights = np.random.rand(i,o,h,w).astype("float32")
            m1 = jt.nn.ConvTranspose(i,o,h, stride=stride, dilation=dilation, bias=False)
            m2 = torch.nn.ConvTranspose2d(i,o,h, stride=stride, dilation=dilation, bias=False)
            m1.weight.data = weights
            m2.weight.data = torch.Tensor(weights)
            x = jt.array(img)
            out1 = m1(x)
            mask = jt.random(out1.shape)
            out1 = out1*mask
            tx = torch.Tensor(img)
            tx.requires_grad = True
            out2 = m2(tx) * torch.Tensor(mask.data)
            with jt.log_capture_scope(log_silent=1, 
                log_vprefix="var_re=0,conv=0,op.cc=100") as logs:
                assert np.allclose(out1.data, out2.data)
                dx, dw = jt.grad(out1, [x, m1.weight])
                jt.sync([dx, dw])
                out2.sum().backward()
                assert np.allclose(dw.data, m2.weight.grad.numpy(), 1e-3)
                assert np.allclose(dx.data, tx.grad.numpy())
            assert len(find_log_with_re(logs, "conv")) == 3
        check((4, 5, 10, 10), (5, 6, 3, 3))
        check((4, 5, 10, 10), (5, 6, 3, 3), 2)
        check((4, 5, 100, 100), (5, 6, 4, 4), 2)
        check((4, 5, 100, 100), (5, 6, 4, 4), 3)
        check((4, 5, 100, 100), (5, 6, 5, 5), 1, 2)
        check((4, 5, 100, 100), (5, 6, 5, 5), 2, 2)
        check((4, 5, 100, 100), (5, 6, 5, 5), 2, 3)
        
if __name__ == "__main__":
    unittest.main()
