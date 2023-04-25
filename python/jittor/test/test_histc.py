# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
import unittest

try:
    import torch
    from torch.autograd import Variable
    import autograd.numpy as anp
    from autograd import jacobian

    has_autograd = True
except:
    has_autograd = False

@unittest.skipIf(not has_autograd, "No autograd found.")
class TestHistc(unittest.TestCase):
    def test_histc(self):
        for i in range(30):
            inputs = np.random.uniform(0,10,(40,40))
            tn, tm = np.random.randn(3, 3).astype('float32'), np.random.randn(3, 3).astype('float32')
            x = jt.array(inputs)
            t_x = torch.from_numpy(inputs)
            if i % 2:
                min = max = 0
            else:
                min = (inputs.min() + inputs.max()) / 3
                max = (inputs.min() + inputs.max()) / 3 * 2
            joup = jt.histc(x, bins=i+1, min=min, max=max)
            toup = torch.histc(t_x, bins=i+1, min=min, max=max)
            np.testing.assert_allclose(joup.data, toup.cpu().numpy(), rtol=1e-4, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
