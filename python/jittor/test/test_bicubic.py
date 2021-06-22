# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
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
    from torch.nn import functional as F
except:
    torch = None
    skip_this_test = True


@unittest.skipIf(skip_this_test, "No Torch found")
class TestBicubicInterpolate(unittest.TestCase):
    # this is for testing bicubic interpolate
    def test_bicubic(self):
        for _ in range(20):
          try:
            tn = np.random.randn(1,1,5,5).astype('float32')
            ja = jt.array(tn)
            ta = torch.autograd.Variable(torch.from_numpy(tn),requires_grad=True)
            # test upsample
            ju = jt.nn.interpolate(ja,scale_factor=2,mode='bicubic')
            tu = F.interpolate(ta,scale_factor=2,mode='bicubic')
            assert np.allclose(ju.data,tu.detach().numpy(),rtol=1e-03,atol=1e-06)
            gju = jt.grad(ju,ja)
            gtu = torch.autograd.grad(tu,ta,torch.ones_like(tu),retain_graph=True)[0]
            assert np.allclose(gju.data,gtu.detach().numpy(),rtol=1e-03,atol=1e-06)
            # test align
            je = jt.nn.interpolate(ja,scale_factor=2,mode='bicubic',align_corners=True)
            te = F.interpolate(ta,scale_factor=2,mode='bicubic',align_corners=True)
            assert np.allclose(je.data,te.detach().numpy(),rtol=1e-03,atol=1e-06)
            gje = jt.grad(je,ja)
            gte = torch.autograd.grad(te,ta,torch.ones_like(tu),retain_graph=True)[0]
            assert np.allclose(gje.data,gte.detach().numpy(),rtol=1e-03,atol=1e-06)
          except AssertionError:
            print(ju,tu)
            print(je,te)

if __name__ == "__main__":
    unittest.main()