"""A float64 mean VJP must not inherit a float32 reciprocal coefficient."""

import os

import jittor as jt
import numpy as np
import pytest

from jittor._runtime import forbid_backend_fallbacks


@pytest.mark.parametrize("length", [3, 129, 257])
def test_float64_mean_backward_keeps_the_reciprocal_precision(length):
    device = os.environ.get("JITTOR_TEST_DEVICE", "cpu")
    source = np.linspace(-1., 1., 2 * length).reshape(2, length)
    seed = np.array([1.25, -2.5], dtype="float64")
    with jt.flag_scope(use_cuda=int(device == "cuda"), amp_reg=0,
                       auto_convert_64_to_32=0), forbid_backend_fallbacks():
        x = jt.array(source, dtype="float64")
        gradient = jt.grad((x.mean(-1) * jt.array(seed, dtype="float64")).sum(), x)
        gradient.sync()
        assert str(gradient.dtype) == "float64"
        assert gradient.location() == ("device" if device == "cuda" else "cpu")
        expected = np.broadcast_to((seed / length)[:, None], source.shape)
        np.testing.assert_allclose(gradient.numpy(), expected, rtol=2e-15, atol=0.)
