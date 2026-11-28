"""CUDA pointwise convolution for spatially pooled activations."""

import jittor as jt
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", [F.silu, F.sigmoid, None])
def test_single_pixel_conv_bias_and_activation(dtype, activation):
    rng = np.random.default_rng(61)
    with torch.no_grad(), forbid_backend_fallbacks():
        def tensor(shape):
            values = rng.integers(-4, 5, size=shape).astype(np.float32) / 8
            return torch.tensor(values, dtype=dtype, device="cuda")

        image = tensor((3, 53, 1, 1))
        weight = tensor((37, 53, 1, 1))
        bias = tensor((37,))
        jt.sync_all(True)
        ref = F.conv2d(image.expand(3, 53, 2, 2).contiguous(), weight, bias)[..., :1, :1]
        if activation is not None:
            ref = activation(ref)
        ref.sync()
        with jt.profile(device=True, memory=False) as trace:
            out = F.conv2d(image, weight, bias)
            if activation is not None:
                out = activation(out)
            out.sync()
        kernels = [row["name"] for row in trace.result.kernel_records]
        assert len(kernels) == 1 and "jt_pointwise_pixel" in kernels[0]
        assert out.location() == "device"
        np.testing.assert_array_equal(out.float().cpu().numpy(), ref.float().cpu().numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_single_pixel_conv_sigmoid_gates_channels_last_features(dtype):
    rng = np.random.default_rng(87)
    with torch.no_grad(), forbid_backend_fallbacks():
        def tensor(shape):
            values = rng.integers(-4, 5, size=shape).astype(np.float32) / 8
            return torch.tensor(values, dtype=dtype, device="cuda")

        feature = tensor((2, 7, 11, 37)).permute(0, 3, 1, 2)
        image = tensor((2, 53, 1, 1))
        weight = tensor((37, 53, 1, 1))
        bias = tensor((37,))
        jt.sync_all(True)
        gate = F.sigmoid(F.conv2d(image, weight, bias))
        gate.sync()
        ref = feature * gate
        ref.sync()
        with jt.profile(device=True, memory=False) as trace:
            result = feature * F.sigmoid(F.conv2d(image, weight, bias))
            result.sync()
        kernels = [row["name"] for row in trace.result.kernel_records]
        assert len(kernels) == 1 and "jt_pointwise_sigmoid_gate" in kernels[0]
        assert result.location() == "device" and result.stride() == feature.stride()
        np.testing.assert_array_equal(result.float().cpu().numpy(), ref.float().cpu().numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_single_pixel_conv_module_replays_the_fused_kernel():
    with torch.no_grad(), forbid_backend_fallbacks(), jt.flag_scope(auto_graph_replay=0):
        conv = torch.nn.Conv2d(24, 384, 1).to(device="cuda", dtype=torch.float16)
        x = torch.ones((1, 24, 1, 1), dtype=torch.float16, device="cuda")
        jt.sync([x, conv.weight, conv.bias], True)
        forward = torch.compile(lambda x: F.silu(conv(x)), mode="reduce-overhead")
        for _ in range(6):
            forward(x).sync()
        with jt.profile(device=True, memory=False) as trace:
            out = forward(x)
            out.sync()
        assert out.location() == "device" and forward.stats["graph"] > 0
        assert sum("jt_pointwise_pixel" in row["name"] for row in trace.result.kernel_records) == 1
