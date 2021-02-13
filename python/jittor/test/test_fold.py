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


class TestFoldOp(unittest.TestCase):
    def test_fold(self):
        # test unfold first and the test fold.
        for _ in range(100):
          # test unfold
          tn = np.random.randn(1,3,4,4).astype('float32')
          ja = jt.array(tn)
          ta = torch.from_numpy(tn)
          juf = jt.nn.unfold(ja,kernel_size=2,stride=2,dilation=2,padding=2)
          tuf = F.unfold(ta,kernel_size=2,stride=2,dilation=2,padding=2)
          assert np.allclose(juf.data,tuf.numpy())
          # test fold
          jf = jt.nn.fold(juf,output_size=(4,4),kernel_size=2,stride=2,dilation=2,padding=2)
          tf = F.fold(tuf,output_size=(4,4),kernel_size=2,stride=2,dilation=2,padding=2)
          assert np.allclose(jf.data,tf.numpy())

if __name__ == "__main__":
    unittest.main()
