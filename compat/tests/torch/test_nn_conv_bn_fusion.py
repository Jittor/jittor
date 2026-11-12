import pytest
import torch
from torch import nn
from _helpers.common import get_all_device_types


@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('affine', [False, True])
@pytest.mark.parametrize('device', get_all_device_types())
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


@pytest.mark.parametrize('device', get_all_device_types())
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('has_bias', [False, True])
def test_fused_weights_preserve_frontend_metadata_and_frozen_state(device, transpose, has_bias):
    import numpy as np
    from torch.nn.utils.fusion import fuse_conv_bn_weights
    shape = (2, 3, 1) if transpose else (3, 2, 1)
    weight = torch.arange(6, dtype=torch.float64, device=device).reshape(shape)
    bias = torch.tensor([1., 2., 3.], device=device, requires_grad=True) if has_bias else None
    mean = torch.tensor([1., 2., 3.], device=device)
    variance = torch.tensor([4., 9., 16.], device=device)
    scale = torch.tensor([4., 3., 2.], device=device)
    offset = torch.tensor([.5, -.5, 1.], device=device)
    fused_w, fused_b = fuse_conv_bn_weights(weight, bias, mean, variance, 0., scale, offset, transpose)
    for value in (fused_w, fused_b):
        assert isinstance(value, torch.nn.Parameter)
        assert value.device == weight.device
        assert value.is_leaf
    assert fused_w.dtype == torch.float64
    assert fused_b.dtype == (torch.float32 if has_bias else torch.float64)
    assert not fused_w.requires_grad
    assert fused_b.requires_grad == has_bias
    scales = np.array([2., 1., .5]).reshape((1, 3, 1) if transpose else (3, 1, 1))
    expected_w = np.arange(6).reshape(shape) * scales
    expected_b = np.array([.5, -.5, 1.]) if has_bias else np.array([-1.5, -2.5, -.5])
    np.testing.assert_allclose(fused_w.detach().cpu().numpy(), expected_w, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(fused_b.detach().cpu().numpy(), expected_b, rtol=1e-6, atol=1e-7)
