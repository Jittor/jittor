"""Preserve the GELU rounding when a CUDA gate and its product share a pass."""

import jittor as jt
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("adjacent", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_gate_cuda(adjacent, dtype):
    rng = np.random.default_rng(83)
    width = 2048 if not adjacent else 769
    with torch.no_grad(), forbid_backend_fallbacks():
        rows = 6 if adjacent else 65
        packed = torch.tensor(rng.normal(size=(8, rows, 2 * width)), dtype=dtype, device="cuda")
        packed.sync()
        gate, value = packed.chunk(2, dim=-1)
        if not adjacent:
            gate, value = gate.contiguous(), value.contiguous()
            jt.sync([gate, value], True)
        expected = F.gelu(gate, approximate="tanh")
        expected.sync()
        expected = expected * value
        expected.sync()
        output = F.gelu(gate, approximate="tanh") * value
        assert output._producer_name() == "code"
        output.sync()
        assert output.location() == "device"
        np.testing.assert_array_equal(output.float().cpu().numpy(), expected.float().cpu().numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gelu_gate_cuda_consumes_pending_linear_output():
    rng = np.random.default_rng(91)
    dtype = torch.bfloat16
    with torch.no_grad(), forbid_backend_fallbacks():
        def tensor(shape):
            return torch.tensor(rng.normal(size=shape), device="cuda", dtype=dtype)

        x = tensor((16, 256))
        weight = tensor((384, 256))
        value = tensor((16, 384))
        jt.sync([x, weight, value], True)
        gate = F.linear(x, weight)
        assert not gate.is_finished
        actual = F.gelu(gate, approximate="tanh") * value
        assert actual._producer_name() == "code"
        actual.sync()
        assert actual.location() == "device"
        activated = F.gelu(gate, approximate="tanh")
        activated.sync()
        expected = activated * value
        np.testing.assert_array_equal(actual.float().cpu().numpy(), expected.float().cpu().numpy())
