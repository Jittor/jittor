"""Fuse frozen normalization into channels-last depthwise convolution."""

import jittor as jt
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("silu", [False, True])
def test_depthwise_batch_norm_keeps_layout_and_rounding(dtype, silu):
    rng = np.random.default_rng(49)
    with torch.no_grad(), forbid_backend_fallbacks():
        def tensor(shape):
            return torch.tensor(rng.normal(size=shape), dtype=dtype, device="cuda")

        x = tensor((2, 11, 13, 32)).permute(0, 3, 1, 2)
        weight = tensor((32, 1, 3, 3))
        mean, variance, gain, bias = [tensor((32,)) for _ in range(4)]
        variance = variance.abs() + 0.2
        jt.sync_all(True)

        def conv():
            return F.conv2d(x, weight, padding=1, groups=32)

        def norm(value):
            return F.batch_norm(value, mean, variance, gain, bias, training=False)

        separate = conv()
        separate.sync()
        expected = norm(separate)
        if silu:
            expected = F.silu(expected)
        expected.sync()

        with jt.profile(device=True, memory=False) as profile:
            actual = norm(conv())
            if silu:
                actual = F.silu(actual)
            actual.sync()
        assert actual.location() == "device"
        assert actual.stride() == separate.stride()
        kernels = [row["name"] for row in profile.result.kernel_records]
        assert len(kernels) == 1 and "KernelDepthwiseConvNHWC" in kernels[0]
        np.testing.assert_array_equal(actual.float().cpu().numpy(), expected.float().cpu().numpy())
