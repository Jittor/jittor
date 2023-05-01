# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
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
    has_autograd = True
except:
    has_autograd = False

@unittest.skipIf(not has_autograd, "No autograd found.")
class TestDigamma(unittest.TestCase):
    def test_digamma(self):
        for i in range(30):
            nx = np.random.uniform(0, 1, (32, 32))
            x = jt.array(nx)
            tx = torch.autograd.Variable(torch.tensor(nx, dtype=torch.float32), requires_grad=True)
            dx = jt.digamma.apply(x)
            tdx = torch.digamma(tx)
            np.testing.assert_allclose(dx.data, tdx.detach().numpy(), rtol=1e-4, atol=1e-6)
            jgdx = jt.grad(dx, x)
            tgdx = torch.autograd.grad(tdx, tx, torch.ones_like(tx))[0]
            np.testing.assert_allclose(jgdx.data, tgdx.detach().numpy(), rtol=1e-4, atol=1e-6)

@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestCudaDigamma(TestDigamma):
    def setUp(self):
        jt.flags.use_cuda = 1
    def tearDown(self):
        jt.flags.use_cuda = 0

if __name__ == "__main__":
    unittest.main()
