"""Real ACL GELU algorithms, gradients and alternating attribute programs."""

import math

import jittor as jt
import numpy as np
import pytest

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels.ops import gelu_op


@pytest.fixture(autouse=True)
def acl_device():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, transpose_storage_view=1), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


def reference(source, approximate):
    x = source.astype(np.float64)
    if approximate == "none":
        cdf = .5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
        return x * cdf, cdf + x * np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    coefficient = math.sqrt(2 / math.pi)
    t = np.tanh(coefficient * (x + .044715 * x ** 3))
    return .5 * x * (1 + t), .5 * (1 + t) + .5 * x * (1 - t * t) * coefficient * (1 + .134145 * x * x)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("strided", [False, True])
def test_public_forward_and_gradient_match_independent_formula(approximate, dtype, strided, monkeypatch):
    rng = np.random.RandomState(20261022)
    source = rng.uniform(-5, 5, (3, 17, 11)).astype(dtype)
    source.flat[:16] = [-8, -4, -3, -2.5, -2, -1.5, -1, -.5, -.01, 0, .01, .5, 1, 2, 4, 8]
    seed = rng.uniform(-1, 2, source.shape).astype(dtype)
    if strided:
        x = jt.core.ops.transpose(jt.array(source.transpose(0, 2, 1).copy()), (0, 2, 1))
        assert not x._storage_is_contiguous()
    else:
        x = jt.array(source)
    x.start_grad()
    calls = []
    original = gelu_op.GeluACL.__call__

    def record(self, value, approximate="none"):
        calls.append(approximate)
        return original(self, value, approximate)

    monkeypatch.setattr(gelu_op.GeluACL, "__call__", record)
    output = jt.nn.gelu(x, approximate=approximate)
    gradient = jt.grad((output * jt.array(seed)).sum(), x)
    jt.sync([output, gradient], True)
    assert calls == [approximate]
    for value in (output, gradient):
        assert str(value.dtype) == dtype
        assert tuple(value.shape) == source.shape
        assert value.location() == "device"
    expected, derivative = reference(source, approximate)
    actual, actual_gradient = jt.fetch_sync([output, gradient])
    assert np.isfinite(actual).all() and np.isfinite(actual_gradient).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-3 if dtype == "float16" else 3e-5,
                               atol=1e-3 if dtype == "float16" else 3e-6)
    np.testing.assert_allclose(actual_gradient, derivative * seed,
                               rtol=3e-3 if dtype == "float16" else 4e-5,
                               atol=2e-3 if dtype == "float16" else 4e-6)


def test_alternating_modes_preserve_forward_and_backward_program_attributes():
    source = np.array([-3, -2.5, -2, -1, 0, 1, 2, 2.5, 3], dtype=np.float32)
    for approximate in ("none", "tanh", "none", "tanh"):
        x = jt.array(source)
        output = jt.nn.gelu(x, approximate=approximate)
        gradient = jt.grad(output.sum(), x)
        expected, derivative = reference(source, approximate)
        jt.sync([output, gradient], True)
        actual, actual_gradient = jt.fetch_sync([output, gradient])
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(actual_gradient, derivative, rtol=4e-5, atol=4e-6)
    exact = gelu_op._gelu_program("none")
    tanh = gelu_op._gelu_program("tanh")
    assert exact is gelu_op._gelu_program("none")
    assert tanh is gelu_op._gelu_program("tanh")
    assert exact is not tanh and exact.data != tanh.data


def test_unknown_algorithm_is_rejected_before_backend_execution():
    with pytest.raises(ValueError, match="approximate"):
        jt.nn.gelu(jt.ones((3,)), approximate="fast")


def test_siglip_activation_shape_preserves_dtype_and_device():
    source = np.random.RandomState(311).normal(size=(1, 196, 3072)).astype(np.float16)
    with jt.no_grad():
        output = jt.nn.gelu(jt.array(source), approximate="tanh")
        output.sync(True)
        assert str(output.dtype) == "float16" and output.location() == "device"
        expected, _ = reference(source, "tanh")
        np.testing.assert_allclose(output.numpy(), expected, rtol=2e-3, atol=1e-3)
