"""Short-row opmath, nonfinite masking and connected second derivatives."""

import os
import jittor as jt
import numpy as np
import pytest
from jittor._runtime import forbid_backend_fallbacks
from jittor.backends.cuda.kernels.nn.softmax_cuda import softmax_v1


@pytest.mark.parametrize("length", [1, 7, 31, 129, 1024])
@pytest.mark.parametrize("log", [False, True])
def test_short_softmax_forward_vjp_and_hessian(length, log):
    if os.environ.get("JITTOR_TEST_DEVICE", "cpu") != "cuda":
        pytest.skip("CUDA short-row implementation")
    rng = np.random.RandomState(629)
    raw = rng.randn(3, length).astype("float32")
    seed = rng.randn(3, length).astype("float32")
    direction = rng.randn(3, length).astype("float32")
    shifted = raw.astype("float64") - raw.max(-1, keepdims=True)
    prob = np.exp(shifted)
    prob /= prob.sum(-1, keepdims=True)
    expected = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True)) if log else prob
    gradient = (
        seed - prob * seed.sum(-1, keepdims=True)
        if log
        else prob * (seed - (seed * prob).sum(-1, keepdims=True))
    )
    if log:
        hv = (
            -seed.sum(-1, keepdims=True)
            * prob
            * (direction - (direction * prob).sum(-1, keepdims=True))
        )
    else:
        py = (prob * direction).sum(-1, keepdims=True)
        gy = (prob * seed).sum(-1, keepdims=True)
        hv = prob * (
            (direction - py) * (seed - gy) - (prob * direction * (seed - gy)).sum(-1, keepdims=True)
        )
    with jt.flag_scope(use_cuda=1, amp_reg=0), forbid_backend_fallbacks():
        x = jt.array(raw)
        y = softmax_v1(x, log=log)
        dx = jt.grad((y * jt.array(seed)).sum(), x)
        hvp = jt.grad((dx * jt.array(direction)).sum(), x)
        for value, reference in [(y, expected), (dx, gradient), (hvp, hv)]:
            value.sync()
            assert value.location() == "device"
            np.testing.assert_allclose(value.numpy(), reference, rtol=2e-05, atol=2e-06)


def test_short_softmax_negative_infinity_and_zero_mask():
    if os.environ.get("JITTOR_TEST_DEVICE", "cpu") != "cuda":
        pytest.skip("CUDA short-row implementation")
    raw = np.array([[0.0, -np.inf, 1.0, -np.inf, -2.0, 0.0, 2.0], [-np.inf] * 7], dtype="float32")
    with jt.flag_scope(use_cuda=1, amp_reg=0), forbid_backend_fallbacks():
        x = jt.array(raw)
        ordinary = softmax_v1(x)
        zero = softmax_v1(x, zero_all_neg_inf=True)
        logged = softmax_v1(x, log=True)
        assert np.isnan(ordinary.numpy()[1]).all()
        assert np.isnan(logged.numpy()[1]).all()
        np.testing.assert_array_equal(zero.numpy()[1], np.zeros(7))
        assert ordinary.numpy()[0, 1] == 0.0
        assert np.isneginf(logged.numpy()[0, 1])
        np.testing.assert_allclose(ordinary.numpy()[0].sum(), 1.0, rtol=1e-06)
