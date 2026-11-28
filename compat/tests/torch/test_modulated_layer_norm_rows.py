"""Modulated LayerNorm keeps the low-precision rounding and wide statistics."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device,amplitude", [("cpu", 4.0), ("cuda", 4.0), ("cuda", 1e20)])
def test_modulation_after_layer_norm_with_many_rows(device, amplitude):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    width = 1176
    signs = np.tile(np.array([-1, 1], dtype=np.float32), width // 2)
    values = np.broadcast_to(signs * amplitude, (2, 257, width)).copy()
    with torch.no_grad(), forbid_backend_fallbacks():
        x = torch.tensor(values, dtype=torch.bfloat16, device=device)
        scale = torch.full((2, 1, width), 0.5, dtype=x.dtype, device=device)
        shift = torch.full((2, 1, width), 0.25, dtype=x.dtype, device=device)
        y = F.layer_norm(x, (width,), eps=1e-5) * (1 + scale) + shift
        assert y.device.type == device
        assert y.dtype == x.dtype
        if device == "cuda":
            assert y._producer_name() == "code"
        expected = np.broadcast_to(signs * 1.5 + 0.25, values.shape)
        np.testing.assert_array_equal(y.float().cpu().numpy(), expected)
