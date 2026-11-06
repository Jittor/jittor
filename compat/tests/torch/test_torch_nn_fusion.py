"""torch.nn.utils.fusion.fuse_conv_bn_eval matches the unfused eval model."""

from __future__ import print_function

import unittest

import torch
from torch.nn.utils.fusion import fuse_conv_bn_eval


class TestConvBnFusion(unittest.TestCase):
    def test_fused_conv_matches_conv_bn_in_eval(self):
        torch.manual_seed(0)
        conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        bn = torch.nn.BatchNorm2d(4)
        bn.running_mean = torch.tensor([0.1, -0.2, 0.3, 0.0])
        bn.running_var = torch.tensor([1.0, 0.5, 2.0, 1.5])
        bn.weight = torch.nn.Parameter(torch.tensor([1.0, 0.8, 1.2, 0.9]))
        bn.bias = torch.nn.Parameter(torch.tensor([0.0, 0.1, -0.1, 0.2]))
        conv.eval(); bn.eval()
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            expected = bn(conv(x))
            fused = fuse_conv_bn_eval(conv, bn)
            actual = fused(x)
        delta = (expected - actual).abs().max().item()
        self.assertLess(delta, 1e-5)


if __name__ == '__main__':
    unittest.main()
