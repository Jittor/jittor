# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved.
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

cupy = None
try:
    import cupy
except:
    pass

@unittest.skipIf(not has_autograd, "No autograd found.")
class TestEinsum(unittest.TestCase):
    def test_einsum_ijjk(self):
        for i in range(30):
            string = "ij,jk->ik"
            tn, tm = np.random.randn(3, 3).astype('float32'), np.random.randn(3, 3).astype('float32')
            x = jt.array(tn)
            y = jt.array(tm)
            t_x = torch.from_numpy(tn)
            t_y = torch.from_numpy(tm)
            t_x = Variable(t_x, requires_grad=True)
            t_y = Variable(t_y, requires_grad=True)
            jq = jt.linalg.einsum(string, x, y)
            tq = torch.einsum(string, t_x, t_y)
            np.testing.assert_allclose(jq.data, tq.detach().numpy(), rtol=1e-4, atol=1e-6)
            gq = jt.grad(jq, x).data
            gr = jt.grad(jq, y).data
            tgq = torch.autograd.grad(tq, t_x, torch.ones_like(tq), retain_graph=True)
            tgr = torch.autograd.grad(tq, t_y, torch.ones_like(tq))
            np.testing.assert_allclose(gq, tgq[0].numpy(), rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(gr, tgr[0].numpy(), rtol=1e-4, atol=1e-6)
    
    def test_einsum_ii(self):
        for i in range(30):
            string = "ij->i"
            tn, tm = np.random.randn(3, 3).astype('float32'), np.random.randn(3, 3).astype('float32')
            x = jt.array(tn)
            # x = x.reindex([2, 2, x.shape[0], x.shape[1]], ["i2", "i3"])
            t_x = torch.from_numpy(tn)
            t_x = Variable(t_x, requires_grad=True)
            jq = jt.linalg.einsum(string, x)
            tq = torch.einsum(string, t_x)
            np.testing.assert_allclose(jq.data, tq.detach().numpy(), rtol=1e-4, atol=1e-6)
            gq = jt.grad(jq, x).data
            tgq = torch.autograd.grad(tq, t_x, torch.ones_like(tq))
            np.testing.assert_allclose(gq, tgq[0].numpy(), rtol=1e-4, atol=1e-6)
    
    def test_einsum_multi(self):
       for i in range(30):
            string = "ij,ijk,jk->ik"
            tn, tm, tk = np.random.randn(3, 4).astype('float32'), np.random.randn(3, 4, 5).astype('float32'), np.random.randn(4, 5).astype('float32')
            x = jt.array(tn)
            y = jt.array(tm)
            z = jt.array(tk)
            # x = x.reindex([2, 2, x.shape[0], x.shape[1]], ["i2", "i3"])
            t_x = torch.from_numpy(tn)
            t_y = torch.from_numpy(tm)
            t_z = torch.from_numpy(tk)
            t_x = Variable(t_x, requires_grad=True)
            t_y = Variable(t_y, requires_grad=True)
            t_z = Variable(t_z, requires_grad=True)
            jq = jt.linalg.einsum(string, x, y, z)
            tq = torch.einsum(string, t_x, t_y, t_z)
            np.testing.assert_allclose(jq.data, tq.detach().numpy(), rtol=1e-4, atol=1e-6)
            gq = jt.grad(jq, x).data
            gr = jt.grad(jq, y).data
            gz = jt.grad(jq, z).data
            tgq = torch.autograd.grad(tq, t_x, torch.ones_like(tq), retain_graph=True)
            tgr = torch.autograd.grad(tq, t_y, torch.ones_like(tq), retain_graph=True)
            tgz = torch.autograd.grad(tq, t_z, torch.ones_like(tq), retain_graph=True)
            np.testing.assert_allclose(gq, tgq[0].numpy(), rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(gr, tgr[0].numpy(), rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(gz, tgz[0].numpy(), rtol=1e-4, atol=1e-6)


@unittest.skipIf(not jt.compiler.has_cuda or cupy is None, "No CUDA found")
class TestCudaEinsum(TestEinsum):
    def setUp(self):
        jt.flags.use_cuda = 1
    def tearDown(self):
        jt.flags.use_cuda = 0

if __name__ == "__main__":
    unittest.main()
