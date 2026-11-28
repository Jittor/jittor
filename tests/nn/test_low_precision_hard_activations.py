"""Low-precision Hardswish and Hardsigmoid use FP32 opmath."""

import jittor as jt
import numpy as np
import pytest

from _helpers.capability import require_accelerator


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_hard_activations_preserve_channels_last(backend, dtype):
    if backend == "cuda":
        require_accelerator("cuda")
    values = np.array([-12000, -3.25, -3, -2.988, -0.375, -0.0, 0.625, 1.125,
                       2.99, 3, 3.25, 12000, -np.inf, np.inf, np.nan, -2.8],
                      dtype="float32")
    features = np.tile(values, (1, 5, 7, 1))
    with jt.flag_scope(use_cuda=int(backend == "cuda"), backend_fallback="error"), jt.no_grad():
        x = jt.array(features).cast(dtype)._storage_permute((0, 3, 1, 2))
        wide = x.float32().numpy()
        gate = np.clip(wide + 3, 0, 6)
        with np.errstate(invalid="ignore"):
            swish = wide * gate / 6
        for output, expected in ((jt.nn.hardsigmoid(x), gate / 6),
                                 (jt.nn.hardswish(x), swish)):
            assert tuple(output._storage_strides()) == tuple(x._storage_strides())
            assert str(output.dtype) == dtype
            output.sync()
            assert output.location() == ("device" if backend == "cuda" else "cpu")
            expected = jt.array(expected).cast(dtype).float32().numpy()
            np.testing.assert_array_equal(output.float32().numpy(), expected)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_hardswish_low_precision_backward(backend):
    if backend == "cuda":
        require_accelerator("cuda")
    values = np.array([-4, -3, -1.5, 0, 1.5, 3, 4], dtype="float32")
    expected = np.array([0, -0.5, 0, 0.5, 1, 1.5, 1], dtype="float16")
    with jt.flag_scope(use_cuda=int(backend == "cuda"), backend_fallback="error"):
        x = jt.array(values).float16().start_grad()
        gradient = jt.grad(jt.nn.hardswish(x).sum(), x)
        gradient.sync()
        assert gradient.location() == ("device" if backend == "cuda" else "cpu")
        np.testing.assert_array_equal(gradient.numpy(), expected)
