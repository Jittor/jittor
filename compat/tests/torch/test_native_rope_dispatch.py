"""Native recognition of the rotary embedding expression's operand graph."""

import numpy as np
import pytest
import torch

import jittor as jt
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
