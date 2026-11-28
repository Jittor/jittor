"""Fuse a gated activation over adjacent CUDA columns."""

import jittor as jt
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor._runtime.step_capture import tracing


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("offset,width", [(16, 64), (3, 61)])
def test_split_silu_replays_pitched_rows(device, offset, width):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(41)
    with torch.no_grad(), forbid_backend_fallbacks(), jt.flag_scope(auto_graph_replay=0):
        def gated(packed):
            gate, value = packed[..., offset:offset + 2 * width].chunk(2, dim=-1)
            actual = value * F.silu(gate)
            if device == "cuda" and tracing():
                assert actual._producer_name() == "code"
            return actual

        run = torch.compile(gated, mode="reduce-overhead")
        for _ in range(7):
            packed = torch.tensor(rng.normal(size=(2, 9, 144)), dtype=torch.bfloat16, device=device)
            gate = packed[..., offset:offset + width].contiguous()
            value = packed[..., offset + width:offset + 2 * width].contiguous()
            expected = F.silu(gate) * value
            result = run(packed)
            assert result.device.type == device
            assert result.dtype == torch.bfloat16
            np.testing.assert_array_equal(result.float().cpu().numpy(), expected.float().cpu().numpy())
        if device == "cuda":
            assert run.stats["graph"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_modified_activation_is_used_by_the_gate():
    with torch.no_grad(), forbid_backend_fallbacks(), jt.flag_scope(auto_graph_replay=0):
        def gated(packed):
            gate, value = packed.chunk(2, dim=-1)
            activation = F.silu(gate)
            activation.add_(2)
            return activation * value

        run = torch.compile(gated, mode="reduce-overhead")
        for i in range(7):
            packed = torch.full((2, 8, 128), float(i % 2), dtype=torch.float16, device="cuda")
            expected = gated(packed)
            actual = run(packed)
            np.testing.assert_array_equal(actual.cpu().numpy(), expected.cpu().numpy())
        assert run.stats["graph"] > 0
