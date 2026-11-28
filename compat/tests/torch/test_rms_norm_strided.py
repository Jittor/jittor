"""Single-round RMS normalization over projected and transposed rows."""

import jittor as jt
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("start,step,width", [(32, 1, 64), (1, 1, 32), (3, 2, 32), (0, 1, 0)])
def test_rms_norm_on_a_view(device, dtype, start, step, width):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(15)
    eps = 1e-5
    with torch.no_grad(), forbid_backend_fallbacks():
        packed = torch.tensor(rng.normal(size=(2, 7, 128)), device=device, dtype=dtype)
        weight = torch.tensor(rng.uniform(0.4, 1.6, size=width), device=device, dtype=dtype)
        # Keep the channel stride arbitrary and transpose the leading axes.
        view = packed[..., start:start + width * step:step].transpose(0, 1)
        direct = F.rms_norm(view, (width,), weight, eps)
        assert direct.device.type == device
        assert direct.dtype == dtype
        if device == "cuda" and step == 1 and width:
            assert direct._producer_name() == "code"
        reference = F.rms_norm(view.contiguous(), (width,), weight, eps)
        np.testing.assert_array_equal(direct.float().cpu().numpy(), reference.float().cpu().numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rms_norm_view_replays_new_input_values():
    rng = np.random.default_rng(16)
    with torch.no_grad(), forbid_backend_fallbacks(), jt.flag_scope(auto_graph_replay=0):
        weight = torch.ones(32, dtype=torch.bfloat16, device="cuda")

        @torch.compile(mode="reduce-overhead")
        def normalize(packed):
            return F.rms_norm(packed[..., 32:64], (32,), weight, 1e-6)

        for _ in range(8):
            value = torch.tensor(rng.normal(size=(3, 5, 96)), dtype=weight.dtype, device="cuda")
            expected = F.rms_norm(value[..., 32:64].contiguous(), (32,), weight, 1e-6)
            actual = normalize(value)
            assert actual.device.type == "cuda"
            np.testing.assert_array_equal(actual.float().cpu().numpy(), expected.float().cpu().numpy())
        assert normalize.stats["graph"] > 0
