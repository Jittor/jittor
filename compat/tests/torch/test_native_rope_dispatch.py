"""Native recognition of the rotary embedding expression's operand graph."""

import numpy as np
import pytest
import torch

import jittor as jt
from _helpers.capability import require_accelerator
from jittor._runtime.step_capture import tracing
from jittor.ops.concatenation import _detect_rotate_half_source


@pytest.mark.skipif(not jt.compiler.has_cuda, reason="requires CUDA")
def test_rotary_operand_recognition():
    values = np.arange(48, dtype=np.float32).reshape(1, 2, 3, 8)
    factors = np.arange(8, dtype=np.float32) / 8
    with torch.no_grad(), jt.flag_scope(auto_flush_ops=0, async_flush_ops=0):
        x = torch.tensor(values, device="cuda")
        cosine = torch.tensor(factors.reshape(1, 1, 1, 8), device="cuda")
        upper, lower = x[..., 4:], x[..., :4]
        match = _detect_rotate_half_source((-upper, lower), 3)
        assert match is not None and match.var_ptr == x.var_ptr
        assert _detect_rotate_half_source((-lower, upper), 3) is None
        assert _detect_rotate_half_source((-upper, (x + 1)[..., :4]), 3) is None
        assert _detect_rotate_half_source((upper, lower), 3) is None

        scaled = cosine * x
        factor = jt.core._neox_rope_cosine(scaled, x)
        assert factor is not None and factor.is_cuda
        assert jt.core._neox_rope_cosine(scaled, x + 1) is None
        assert jt.core._neox_rope_cosine(cosine + x, x) is None

        result = scaled + torch.cat((-upper, lower), dim=-1) * cosine
        assert result.is_cuda
        rotated = np.concatenate((-values[..., 4:], values[..., :4]), axis=-1)
        np.testing.assert_array_equal(result.cpu().numpy(), values * factors + rotated * factors)
        np.testing.assert_array_equal(factor.cpu().numpy(), factors.reshape(1, 1, 1, 8))

        # Pending transposed inputs (e.g. attention Q/K views inside capture) wrap
        # the multiply in FuseTransposeOp rather than TransposeOp.
        raw = torch.tensor(values.transpose(0, 2, 1, 3), device="cuda")
        qt = (raw + 1.0).transpose(1, 2)
        scaled_t = qt * cosine
        factor_t = jt.core._neox_rope_cosine(scaled_t, qt)
        assert factor_t is not None and factor_t.is_cuda
        np.testing.assert_array_equal(factor_t.cpu().numpy(), factors.reshape(1, 1, 1, 8))


@pytest.mark.cuda
def test_capture_rotary_concat_reads_live_strided_inputs():
    require_accelerator("cuda")
    rng = np.random.default_rng(137)
    values = rng.normal(size=(2, 23, 5, 48)).astype("float32")
    cosine = rng.normal(size=(2, 1, 20, 48)).astype("float32")
    sine = rng.normal(size=(2, 1, 20, 48)).astype("float32")
    producers = []

    def forward(x, cos, sin):
        prefix, tail = x.transpose(1, 2).split((3, 20), dim=2)
        lower, upper = tail.chunk(2, dim=-1)
        rotated = tail * cos + torch.cat((-upper, lower), dim=-1) * sin
        output = torch.cat((prefix, rotated), dim=-2)
        if tracing():
            producers.append((output._producer_name(), len(output._inputs())))
        return output

    def inputs(device, scale):
        return [torch.tensor(t, device=device) for t in (values * scale, cosine * scale, sine)]

    with torch.no_grad(), jt.flag_scope(backend_fallback="error", auto_graph_replay=0):
        step = torch.compile(forward, mode="reduce-overhead", dynamic=False)
        for scale in (1, 2):
            arguments = inputs("cuda", scale)
            jt.sync(arguments, device_sync=True)
            for _ in range(6):
                actual = step(*arguments)
                actual.sync()
            expected = forward(*inputs("cpu", scale))
            assert actual.is_cuda
            assert actual.is_contiguous()
            np.testing.assert_allclose(actual.cpu().numpy(), expected.numpy(), rtol=2e-6, atol=2e-6)
        assert producers and all(item == ("code", 4) for item in producers)
        assert step.stats["graph"] > 0
