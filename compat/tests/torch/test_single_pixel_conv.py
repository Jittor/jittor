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
