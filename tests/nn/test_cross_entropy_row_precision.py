"""Bounded CE statistics preserve odd tails, weights and higher derivatives."""

import os
import jittor as jt
import numpy as np
import pytest
from jittor._runtime import forbid_backend_fallbacks


@pytest.mark.parametrize("length", [1025, 2048, 2052])
def test_cross_entropy_row_statistics_vjp_and_hessian(length):
    if os.environ.get("JITTOR_TEST_DEVICE", "cpu") != "cuda":
        pytest.skip("CUDA row-statistics implementation")
    rng = np.random.RandomState(321)
    raw = rng.randn(3, length).astype("float32")
    targets = np.array([1, length - 2, -100], dtype="int32")
    seed = np.array([1.25, -0.5, 2.0], dtype="float32")
    direction = rng.randn(3, length).astype("float32")
    wide = raw.astype("float64")
    top = wide.max(-1, keepdims=True)
    p = np.exp(wide - top)
    p /= p.sum(-1, keepdims=True)
    scale = seed.copy()
    scale[2] = 0
    expected = (
        np.log(np.exp(wide - top).sum(-1)) + top[:, 0] - wide[np.arange(3), np.maximum(targets, 0)]
    )
    expected[2] = 0
    grad = p.copy()
    grad[np.arange(2), targets[:2]] -= 1
    grad *= scale[:, None]
    hvp = scale[:, None] * p * (direction - (p * direction).sum(-1, keepdims=True))
    with jt.flag_scope(
        use_cuda=int(os.environ.get("JITTOR_TEST_DEVICE", "cpu") == "cuda"), amp_reg=0
    ), forbid_backend_fallbacks():
        x = jt.array(raw)
        y = jt.nn.cross_entropy_loss(x, jt.array(targets), ignore_index=-100, reduction="none")
        dx = jt.grad((y * jt.array(seed)).sum(), x)
        hv = jt.grad((dx * jt.array(direction)).sum(), x)
        for value, reference in [(y, expected), (dx, grad), (hv, hvp)]:
            value.sync()
            assert value.location() == ("device" if jt.flags.use_cuda else "cpu")
            np.testing.assert_allclose(value.numpy(), reference, rtol=2e-05, atol=2e-06)
