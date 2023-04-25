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
from jittor import nn
import numpy as np
import unittest
from jittor.weightnorm import weight_norm

try:
    import torch
    from torch.autograd import Variable
    import autograd.numpy as anp
    from autograd import jacobian

    has_autograd = True
except:
    has_autograd = False

class jt_module(jt.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.linear = jt.array(weight)

    def execute(self, x):
        return jt.matmul(self.linear, x)

class torch_module(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.linear = torch.nn.Parameter(torch.from_numpy(weight))
    
    def forward(self, x):
        return torch.matmul(self.linear, x)

@unittest.skipIf(not has_autograd, "No autograd found.")
class TestWeightNorm(unittest.TestCase):
    def test_weightnorm(self):
        for i in range(30):
            weight = np.random.uniform(0,1,(i+10,40))
            jm = jt_module(weight)
            tm = torch_module(weight)
            inp = np.random.uniform(0,1,(40,i+30))
            torch.nn.utils.weight_norm(tm, 'linear', -1)
            weight_norm(jm, 'linear', -1)
            jinp = jt.array(inp)
            tinp = Variable(torch.from_numpy(inp), requires_grad=True)
            joup = jm(jinp)
            toup = tm(tinp)
            np.testing.assert_allclose(joup.data, toup.detach().numpy(), rtol=1e-4, atol=1e-6)
            gq = jt.grad(joup, jinp).data
            tgq = torch.autograd.grad(toup, tinp, torch.ones_like(toup), retain_graph=True)
            np.testing.assert_allclose(gq, tgq[0].numpy(), rtol=1e-4, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
