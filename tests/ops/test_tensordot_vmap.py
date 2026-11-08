"""Native tensordot batching keeps shared-operand gradients outside vmap."""

import jittor as jt
import numpy as np
import pytest

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        capability.require_accelerator("cuda")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


@pytest.mark.parametrize("dtype", ["float32", "complex64", "complex128"])
@pytest.mark.parametrize("contraction", ["full", "matrix"])
def test_tensordot_vmap_mapped_left_shared_right(device, dtype, contraction):
    rng = np.random.default_rng(928)
    a_shape = (3, 2, 4)
    b_shape = (2, 4) if contraction == "full" else (4, 5)
    arrays = []
    for shape in (a_shape, b_shape):
        data = rng.uniform(-0.7, 0.7, size=shape).astype(dtype)
        if dtype.startswith("complex"):
            data.imag = rng.uniform(-0.5, 0.5, size=shape)
        arrays.append(data)
    a_data, b_data = arrays
    a, b = (jt.array(data, dtype=dtype) for data in arrays)
    dims = ([0, 1], [0, 1]) if contraction == "full" else ([1], [0])
    calls = []

    def contract(left, right):
        calls.append((tuple(left.shape), tuple(right.shape)))
        return jt.nn.tensordot(left, right, dims=dims)

    value = jt.vmap(contract, in_dims=(0, None))(a, b)
    assert calls == [(a_shape[1:], b_shape)]
    loss = value.real.sum() if dtype.startswith("complex") else value.sum()
    # Differentiate the batched expression, not vmap(grad(...)).
    # The one shared b must collect cotangents from every mapped example.
    gradient_a, gradient_b = jt.grad(loss, [a, b])

    if contraction == "full":
        expected_value = np.sum(a_data * b_data, axis=(1, 2))
        expected_a = np.broadcast_to(b_data.conj(), a_shape)
        expected_b = np.sum(a_data.conj(), axis=0)
    else:
        expected_value = np.matmul(a_data, b_data)
        expected_a = np.broadcast_to(np.sum(b_data.conj(), axis=1), a_shape)
        expected_b = np.broadcast_to(
            np.sum(a_data.conj(), axis=(0, 1))[:, None], b_shape
        )
    pairs = (
        (value, expected_value),
        (gradient_a, expected_a),
        (gradient_b, expected_b),
    )
    # All device checks precede every numpy() call, including related outputs.
    for actual, expected in pairs:
        assert tuple(actual.shape) == expected.shape
        assert str(actual.dtype) == dtype
        actual.sync()
        assert actual.location() == ("device" if device == "cuda" else "cpu")

    tolerance = 2e-12 if dtype == "complex128" else 2e-6
    for actual, expected in pairs:
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=tolerance, atol=tolerance
        )
