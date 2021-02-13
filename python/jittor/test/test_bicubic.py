# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved.
# Maintainers:
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import torch
from torch.nn import functional as F
import numpy as np


class TestBicubicInterpolate(unittest.TestCase):
    # this is for testing bicubic interpolate
    def test_bicubic(self):
        for _ in range(20):
          try:
            tn = np.random.randn(1,1,5,5).astype('float32')
            ja = jt.array(tn)
            ta = torch.from_numpy(tn)
            # test upsample
            ju = jt.nn.interpolate(ja,scale_factor=2,mode='bicubic')
            tu = F.interpolate(ta,scale_factor=2,mode='bicubic')
            assert np.allclose(ju.data,tu.numpy(),rtol=1e-03,atol=1e-06)
            # test fold
            je = jt.nn.interpolate(ja,scale_factor=2,mode='bicubic',align_corners=True)
            te = F.interpolate(ta,scale_factor=2,mode='bicubic',align_corners=True)
            assert np.allclose(je.data,te.numpy(),rtol=1e-03,atol=1e-06)
          except AssertionError:
            print(ju,tu)
            print(je,te)

if __name__ == "__main__":
    unittest.main()
