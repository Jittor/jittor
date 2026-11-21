"""Antialiased resize semantics and unsupported-mode checks."""

import jittor as jt
import numpy as np
import pytest
import torch

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device", ["cpu", "acl"])
@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_unsupported_antialias_inputs_reject_and_ordinary_interpolation_works(device, mode):
    if device == "acl" and not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=int(device == "acl"), use_acl=int(device == "acl")), forbid_backend_fallbacks(), torch.no_grad():
        x = torch.ones((1, 2, 4, 5), device="cuda" if device == "acl" else "cpu")
        if device == "acl":
            with pytest.raises(NotImplementedError, match="antialias=True.*not implemented"):
                torch.nn.functional.interpolate(x, size=(2, 3), mode=mode, antialias=True)
        else:
            with pytest.raises(ValueError, match="antialias=True.*bilinear or bicubic"):
                torch.nn.functional.interpolate(x, size=(2, 3), mode="nearest", antialias=True)
        # Exercise a supported ordinary path on each backend. The ACL
        # bilinear provider is currently restricted to Ascend 310P.
        ordinary_mode = "nearest" if device == "acl" and mode == "bilinear" else mode
        for kwargs in ({}, {"antialias": False}):
            align = None if ordinary_mode == "nearest" else False
            result = torch.nn.functional.interpolate(x, size=(4, 5), mode=ordinary_mode, align_corners=align, **kwargs)
            jt.sync_all(True)
            assert result.device.type == ("npu" if device == "acl" else "cpu")
            assert bool((result == x).all().item())
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "mode, output, gradient",
    [
        ("bilinear", [1.857142857142857, 4.5, 5.571428571428571],
         [0.428571428571429, 0.678571428571429, 0.892857142857143,
          1.178571428571429, 1.535714285714286, 1.285714285714286]),
        ("bicubic", [1.615062761506276, 4.625954198473282, 5.648535564853557],
         [0.395732856367179, 0.648152288479351, 0.855696445111629,
          1.173688076910792, 1.602127183876840, 1.324603149254208]),
    ],
)
def test_interpolate_antialias_matches_reference_forward_and_backward(device, mode, output, gradient):
    if device == "cuda" and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("requires a real CUDA device")
    with jt.runtime.scope(use_cuda=int(device == "cuda")), forbid_backend_fallbacks():
        x = torch.tensor([[[[0., 4., 1., 9., 2., 8.]]]], device=device,
                         dtype=torch.float64, requires_grad=True)
        result = torch.nn.functional.interpolate(x, size=(1, 3), mode=mode, antialias=True)
        (result * torch.tensor([1., 2., 3.], device=device)).sum().backward()
        assert result.device.type == x.grad.device.type == device
        assert result.dtype == x.grad.dtype == torch.float64
        np.testing.assert_allclose(result.detach().cpu().numpy().ravel(), output, rtol=0, atol=1e-12)
        np.testing.assert_allclose(x.grad.cpu().numpy().ravel(), gradient, rtol=0, atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_interpolate_antialias_preserves_stopped_input(device, mode):
    if device == "cuda" and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("requires a real CUDA device")
    with jt.runtime.scope(use_cuda=int(device == "cuda")), forbid_backend_fallbacks():
        pixels = (torch.arange(3 * 17 * 23, device=device) % 256).to(dtype=torch.uint8)
        image = pixels.reshape(1, 3, 17, 23).to(dtype=torch.float32)
        assert not image.requires_grad
        result = torch.nn.functional.interpolate(image, size=(11, 13), mode=mode, antialias=True)
        assert not result.requires_grad
        assert result.device.type == device
        result.round_()
        assert np.isfinite(result.cpu().numpy()).all()


@pytest.mark.parametrize("mode, expected", [
    ("bilinear", [[88, 102, 85], [114, 111, 126]]),
    ("bicubic", [[86, 102, 82], [113, 108, 131]]),
])
def test_interpolate_antialias_uint8_rounds_each_spatial_pass(mode, expected):
    with jt.runtime.scope(use_cuda=0), forbid_backend_fallbacks():
        x = torch.tensor([[[[0, 128, 255, 30, 120, 10],
                            [150, 40, 10, 80, 20, 250],
                            [255, 12, 120, 220, 190, 30]]]], dtype=torch.uint8, device="cpu")
        result = torch.nn.functional.interpolate(x, size=(2, 3), mode=mode, antialias=True)
        assert result.dtype == torch.uint8 and result.device.type == "cpu"
        np.testing.assert_array_equal(result.numpy(), [[expected]])


def test_interpolate_antialias_cpu_nonsizing_scale_still_weights_the_gradient():
    with jt.runtime.scope(use_cuda=0), forbid_backend_fallbacks():
        x = torch.tensor([[[[1., 5., 2., 3.], [6., 8., 0., 4.]]]],
                         dtype=torch.float64, device="cpu", requires_grad=True)
        result = torch.nn.functional.interpolate(
            x, scale_factor=(1.25, 0.75), mode="bilinear", antialias=True
        )
        result.sum().backward()
        np.testing.assert_allclose(result.detach().numpy(), [[[[2.2, 3.5, 2.7], [6.6, 4., 2.8]]]])
        np.testing.assert_allclose(x.grad.numpy(), [[[[0.91, 1.04, 1.04, 0.91], [0.49, 0.56, 0.56, 0.49]]]])


@pytest.mark.parametrize("device, output, gradient", [
    ("cpu", [1., 4., 7.], [1.25, 7 / 6, 7 / 12]),
    ("cuda", [1., 3.25, 5.75], [1., 1., 1.]),
])
def test_interpolate_antialias_scale_with_unchanged_output_size(device, output, gradient):
    if device == "cuda" and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("requires a real CUDA device")
    with jt.runtime.scope(use_cuda=int(device == "cuda")), forbid_backend_fallbacks():
        x = torch.tensor([[[[1., 4., 7.]]]], dtype=torch.float64, device=device, requires_grad=True)
        result = torch.nn.functional.interpolate(
            x, scale_factor=(1, 1.2), mode="bilinear", antialias=True
        )
        result.sum().backward()
        assert result.data_ptr() != x.data_ptr()
        np.testing.assert_allclose(result.detach().cpu().numpy().ravel(), output, rtol=0, atol=1e-12)
        np.testing.assert_allclose(x.grad.cpu().numpy().ravel(), gradient, rtol=0, atol=1e-12)
