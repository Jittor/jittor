# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Zheng-Ning Liu <lzhengning@gmail.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import unittest
import numpy as np

try:
    import torch
    from emd import earth_mover_distance as TEMD
except:
    skip_this_test = True

import jittor as jt
from jittor.loss3d import chamfer_loss
from jittor.loss3d import earth_mover_distance


class TestLoss3d(unittest.TestCase):
    def test_chamfer(self):
        def test():
            pc1 = np.random.randn(10, 100, 3).astype(np.float32)
            pc2 = np.random.randn(10, 100, 3).astype(np.float32)

            Jpc1 = jt.array(pc1)
            Jpc2 = jt.array(pc2)
            Jcf = chamfer_loss(Jpc1, Jpc2, dims='BNC')

            ppc1 = np.repeat(pc1[:, :, None, :], 100, axis=2)
            ppc2 = np.repeat(pc2[:, None, :, :], 100, axis=1)
            ncf = np.sqrt(((ppc1 - ppc2) ** 2).sum(axis=-1)).min(axis=-1)
            ncf = ncf.mean()

            self.assertTrue(np.allclose(ncf, Jcf.item()))

        test()

        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                test()

    def test_chamfer_dims(self):
        def test():
            pc1 = np.random.randn(10, 100, 3).astype(np.float32)
            pc2 = np.random.randn(10, 100, 3).astype(np.float32)

            Jpc1 = jt.array(pc1.transpose([0, 2, 1]))
            Jpc2 = jt.array(pc2.transpose([0, 2, 1]))
            Jcf = chamfer_loss(Jpc1, Jpc2, dims='BCN')

            ppc1 = np.repeat(pc1[:, :, None, :], 100, axis=2)
            ppc2 = np.repeat(pc2[:, None, :, :], 100, axis=1)
            ncf = np.sqrt(((ppc1 - ppc2) ** 2).sum(axis=-1)).min(axis=-1)
            ncf = ncf.mean()

            self.assertTrue(np.allclose(ncf, Jcf.item()))

        test()
        
        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                test()

    @unittest.skipIf(skip_this_test, "No Pyorch_EMD found")
    def test_emd_torch(self):
        if jt.has_cuda:
            jt.flags.use_cuda = True

        pc1 = np.random.randn(10, 100, 3).astype(np.float32)
        pc2 = np.random.randn(10, 50, 3).astype(np.float32)

        Tpc1 = torch.from_numpy(pc1).cuda()
        Tpc2 = torch.from_numpy(pc2).cuda()
        Tpc1.requires_grad = True
        Tpc2.requires_grad = True
        Temdcost = TEMD(Tpc1, Tpc2, transpose=False)
        Temd = Temdcost.mean()

        Jpc1 = jt.array(pc1)
        Jpc2 = jt.array(pc2)
        Jemd = earth_mover_distance(Jpc1, Jpc2, dims='BNC')

        Temd.backward()
        Tgrad1 = Tpc1.grad.cpu().numpy()
        Tgrad2 = Tpc2.grad.cpu().numpy()

        Jgrad1, Jgrad2 = jt.grad(Jemd, [Jpc1, Jpc2])

        self.assertTrue(np.allclose(Temd.item(), Jemd.item()), Temd.item() - Jemd.item())
        self.assertTrue(np.allclose(Tgrad1, Jgrad1.data, atol=1e-4), np.abs(Tgrad1 - Jgrad1.data).max())
        self.assertTrue(np.allclose(Tgrad2, Jgrad2.data, atol=1e-4), np.abs(Tgrad2 - Jgrad2.data).max())


if __name__ == '__main__':
    unittest.main()