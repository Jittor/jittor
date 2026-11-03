import pytest
import torch
from torch import nn


@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('affine', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fused_conv_bn_eval_preserves_output(bias, affine, device):
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    torch.manual_seed(17)
    conv = nn.Conv2d(2, 3, 3, padding=1, bias=bias).eval()
    bn = nn.BatchNorm2d(3, affine=affine).eval()
    bn.running_mean.copy_(torch.tensor([0.1, -0.2, 0.3]))
    bn.running_var.copy_(torch.tensor([0.4, 1.5, 2.0]))
    if affine:
        bn.weight.data.copy_(torch.tensor([0.5, 0.8, -0.9]))
        bn.bias.data.copy_(torch.tensor([0.3, -0.4, 0.6]))
    conv, bn = conv.to(device), bn.to(device)
    x = torch.randn(2, 2, 5, 5, device=device)
    expected = bn(conv(x)).detach().cpu().numpy()
    fused = fuse_conv_bn_eval(conv, bn)
    actual = fused(x).detach().cpu().numpy()
    import numpy as np
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert fused is not conv


def test_fused_conv_bn_rejects_training():
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    conv = nn.Conv2d(2, 3, 3)
    bn = nn.BatchNorm2d(3)
    with pytest.raises(AssertionError):
        fuse_conv_bn_eval(conv, bn)
