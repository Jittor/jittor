"""The shared rotary entry retains forward and backward math on CPU and CUDA."""

import jittor as jt
import numpy as np
import pytest


@pytest.mark.parametrize("use_cuda", [0, 1])
def test_rotary_emb_matches_split_half_reference_and_gradient(use_cuda):
    if use_cuda and not jt.has_cuda:
        pytest.skip("CUDA is unavailable")
    q = np.arange(32, dtype="float32").reshape(1, 2, 2, 8) / 16
    k = q + 0.25
    cos = np.linspace(0.1, 0.8, 8, dtype="float32").reshape(1, 1, 1, 8)
    sin = np.linspace(-0.4, 0.3, 8, dtype="float32").reshape(1, 1, 1, 8)
    with jt.flag_scope(use_cuda=use_cuda):
        query, key = jt.array(q), jt.array(k)
        outputs = jt.nn.rotary_emb(query, key, freq_cos=jt.array(cos), freq_sin=jt.array(sin))
        gradients = jt.grad(outputs[0].sum() + outputs[1].sum(), [query, key])
        expected_grad = np.concatenate((cos[..., :4] + sin[..., 4:],
                                        cos[..., 4:] - sin[..., :4]), axis=-1)
        for value, result, gradient in zip((q, k), outputs, gradients):
            rotated = np.concatenate((-value[..., 4:], value[..., :4]), axis=-1)
            np.testing.assert_allclose(result.numpy(), value * cos + rotated * sin, atol=1e-6)
            np.testing.assert_allclose(gradient.numpy(), np.broadcast_to(expected_grad, q.shape), atol=1e-6)
