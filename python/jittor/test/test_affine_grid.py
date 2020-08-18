# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor.nn import affine_grid,grid_sample


class TestAffineGrid(unittest.TestCase):
    def test_affine_grid_2d(self):
        import torch.nn.functional as F
        import torch
        N = 8
        C = 3
        H = 256
        W = 128
        theta = np.random.randn(N,2,3).astype(np.float32)
        features = np.random.randint(256,size=(N,C,H,W)).astype(np.float32)

        torch_theta = torch.Tensor(theta)
        torch_features = torch.Tensor(features)
        torch_grid = F.affine_grid(torch_theta,size=(N,C,H,W),align_corners=False)
        torch_sample = F.grid_sample(torch_features,torch_grid,mode='bilinear',padding_mode='zeros',align_corners=False)

        jt_theta = jt.array(theta)
        jt_features = jt.array(features)
        jt_grid = affine_grid(jt_theta,size=(N,C,H,W),align_corners=False)
        jt_sample = grid_sample(jt_features,jt_grid,mode='bilinear',padding_mode='zeros',align_corners=False)

        assert np.allclose(jt_theta.numpy(),torch_theta.numpy())
        assert np.allclose(jt_features.numpy(),torch_features.numpy())
        assert np.allclose(jt_grid.numpy(),torch_grid.numpy(),atol=1e-05)
        assert np.allclose(torch_sample.numpy(),jt_sample.numpy(),atol=1e-01)


    def test_affine_grid_3d(self):
        import torch.nn.functional as F
        import torch
        N = 8
        C = 3
        D = 64
        H = 256
        W = 128
        theta = np.random.randn(N,3,4).astype(np.float32)
        features = np.random.randint(256,size=(N,C,D,H,W)).astype(np.float32)

        torch_theta = torch.Tensor(theta)
        torch_features = torch.Tensor(features)
        torch_grid = F.affine_grid(torch_theta,size=(N,C,D,H,W),align_corners=False)
        torch_sample = F.grid_sample(torch_features,torch_grid,mode='bilinear',padding_mode='zeros',align_corners=False)

        jt_theta = jt.array(theta)
        jt_features = jt.array(features)
        jt_grid = affine_grid(jt_theta,size=(N,C,D,H,W),align_corners=False)
        jt_sample = grid_sample(jt_features,jt_grid,mode='bilinear',padding_mode='zeros',align_corners=False)

        assert np.allclose(jt_theta.numpy(),torch_theta.numpy())
        assert np.allclose(jt_features.numpy(),torch_features.numpy())
        assert np.allclose(jt_grid.numpy(),torch_grid.numpy(),atol=1e-05)
        assert np.allclose(torch_sample.numpy(),jt_sample.numpy(),atol=1e-01)


if __name__ == "__main__":
    unittest.main()