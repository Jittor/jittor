"""Copy noncontiguous CUDA rows directly into a dense concatenate result."""

import jittor as jt
import numpy as np
import pytest
import torch

from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor._runtime.step_capture import tracing


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("channel_offset", [0, 1])
def test_capture_concatenates_transposed_rows(device, channel_offset):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(61)
    channels = slice(channel_offset, channel_offset + 64)
    with torch.no_grad(), forbid_backend_fallbacks(), jt.flag_scope(auto_graph_replay=0):
        def concatenate(a, b):
            result = torch.cat((a[..., channels].transpose(1, 2),
                                b[..., channels].transpose(1, 2)), dim=2)
            if device == "cuda" and not channel_offset and tracing():
                assert result._producer_name() == "code"
            return result

        run = torch.compile(concatenate, mode="reduce-overhead")
        for _ in range(7):
            lhs, rhs = [rng.integers(-32, 32, size=(2, n, 5, 64 + channel_offset)).astype(np.float32)
                        for n in (17, 100)]
            args = [torch.tensor(x, dtype=torch.bfloat16, device=device) for x in (lhs, rhs)]
            result = run(*args)
            assert result.device.type == device
            assert result.dtype == torch.bfloat16
            assert result.is_contiguous()
            expected = np.concatenate((lhs[..., channels].transpose(0, 2, 1, 3),
                                       rhs[..., channels].transpose(0, 2, 1, 3)), axis=2)
            np.testing.assert_array_equal(result.float().cpu().numpy(), expected)
        if device == "cuda":
            assert run.stats["graph"] > 0, run.graph_refused
