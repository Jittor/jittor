"""Numerics and storage behavior of the captured short-attention copy."""
from unittest.mock import patch

import numpy as np
import pytest
import torch
import jittor as jt
from jittor.backends.cuda.kernels.nn import short_attention_cuda

from _rng_init_support import require_rng_device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shifted", [False, True])
def test_capture_short_attention_copy(device, shifted):
    device = require_rng_device(device)
    on_cuda = device == "cuda"
    if on_cuda and min(jt.flags.cuda_archs) < 80:
        pytest.skip("accelerator prerequisite: CUDA compute capability 8.0")
    dtype = torch.float16 if on_cuda else torch.float32
    generator = np.random.default_rng(8961)
    batch, heads, queries, keys, depth, width = 4, 8, 37, 39, 24, 27

    def make_tensor(shape):
        return torch.tensor(generator.normal(0, 0.5, shape).astype(np.float32), dtype=dtype, device=device)

    logits_bias = make_tensor((1, heads, queries, keys))
    shift_values = generator.normal(0, 0.2, (2, 2, 1, queries, keys)).astype(np.float32)
    shift_values[:, :, :, 0, :] = -np.inf
    shift_bias = torch.tensor(shift_values, dtype=dtype, device=device)
    key = make_tensor((batch, keys, heads, depth))
    value = make_tensor((batch, keys, heads, width))

    def forward(query):
        q_rows = query.transpose(1, 2) * 0.5
        logits = (q_rows @ key.permute(0, 2, 3, 1)) + logits_bias
        if shifted:
            logits = (logits.view(2, 2, heads, queries, keys) + shift_bias).view(batch, heads, queries, keys)
        output = logits.softmax(-1) @ value.transpose(1, 2)
        return output.transpose(1, 2).reshape(batch, queries, heads * width)

    with torch.no_grad(), patch.object(short_attention_cuda, "_attention_source",
                                      wraps=short_attention_cuda._attention_source) as source:
        step = jt.capture_step(forward, record=on_cuda)
        try:
            for _ in range(7):
                query = make_tensor((batch, queries, heads, depth))
                expected = forward(query)
                result = step(query)
                jt.sync([expected, result], device_sync=on_cuda)
                assert result.location() == ("device" if on_cuda else "cpu")
                assert result.is_contiguous()
                np.testing.assert_allclose(result.float().cpu().numpy(), expected.float().cpu().numpy(),
                                           atol=8e-4, rtol=4e-3)
                if shifted:
                    assert np.isnan(result.float().cpu().numpy()[:, 0]).all()
            assert step.refused is None
            assert step.stats["replayed"] > 0
            assert bool(source.call_count) == on_cuda
            if on_cuda:
                assert step.stats["graph"] > 0, step.graph_refused
        finally:
            step.invalidate()
