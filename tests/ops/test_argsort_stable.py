"""Native stable argsort contracts on CPU and CUDA, not ACL or native vmap."""

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


def _assert_resident(device, *tensors):
    # Check before numpy() can fetch accelerator storage back to the host.
    for tensor in tensors:
        tensor.sync()
        assert tensor.location() == ("device" if device == "cuda" else "cpu")


def _stable_indices(data, axis, descending):
    # Negating our bounded, finite keys preserves the original order of ties;
    # reversing ascending indices would instead reverse equal-key groups.
    return np.argsort(-data if descending else data, axis=axis, kind="stable")


@pytest.mark.parametrize("key_dtype", ["int32", "float32"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("descending", [False, True])
def test_stable_argsort_long_duplicate_keys(device, key_dtype, index_dtype, descending):
    rng = np.random.default_rng(924)
    data = rng.integers(-3, 4, size=2000).astype(key_dtype)
    if key_dtype == "float32":
        data *= np.float32(0.125)
    x = jt.array(data, dtype=key_dtype)
    indices, values = jt.argsort(
        x, dim=0, descending=descending, dtype=index_dtype, stable=True
    )
    sorted_values, sorted_indices = jt.sort(
        x, dim=0, descending=descending, stable=True
    )
    assert str(indices.dtype) == index_dtype
    assert str(values.dtype) == key_dtype
    _assert_resident(device, indices, values, sorted_values, sorted_indices)
    expected_indices = _stable_indices(data, 0, descending)
    np.testing.assert_array_equal(indices.numpy(), expected_indices)
    np.testing.assert_array_equal(values.numpy(), data[expected_indices])
    np.testing.assert_array_equal(sorted_indices.numpy(), expected_indices)
    np.testing.assert_array_equal(sorted_values.numpy(), data[expected_indices])


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("descending", [False, True])
def test_stable_argsort_nonlast_axis(device, index_dtype, descending):
    rng = np.random.default_rng(925)
    data = rng.integers(-2, 3, size=(2, 2000, 3)).astype("float32")
    indices, values = jt.argsort(
        jt.array(data, dtype="float32"),
        dim=1,
        descending=descending,
        dtype=index_dtype,
        stable=True,
    )
    assert tuple(indices.shape) == data.shape
    assert tuple(values.shape) == data.shape
    assert str(indices.dtype) == index_dtype
    assert str(values.dtype) == "float32"
    _assert_resident(device, indices, values)
    expected_indices = _stable_indices(data, 1, descending)
    np.testing.assert_array_equal(indices.numpy(), expected_indices)
    np.testing.assert_array_equal(
        values.numpy(), np.take_along_axis(data, expected_indices, axis=1)
    )


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_stable_argsort_alternates_jit_modes(device, index_dtype):
    rng = np.random.default_rng(926)
    data = rng.integers(-2, 3, size=2000).astype("float32")
    x = jt.array(data, dtype="float32")
    expected_indices = _stable_indices(data, 0, False)
    expected_values = data[expected_indices]
    for stable in (False, True, False, True):
        indices, values = jt.argsort(
            x, dim=0, descending=False, dtype=index_dtype, stable=stable
        )
        assert str(indices.dtype) == index_dtype
        _assert_resident(device, indices, values)
        actual_indices = indices.numpy()
        actual_values = values.numpy()
        np.testing.assert_array_equal(np.sort(actual_indices), np.arange(data.size))
        np.testing.assert_array_equal(actual_values, data[actual_indices])
        np.testing.assert_array_equal(actual_values, expected_values)
        if stable:
            np.testing.assert_array_equal(actual_indices, expected_indices)
        # stable=False permits any tie order; it must not constrain CUDA's
        # already-stable implementation or an otherwise conforming algorithm.


@pytest.mark.parametrize("descending", [False, True])
def test_stable_argsort_float64_value_gradient(device, descending):
    rng = np.random.default_rng(927)
    # Distinct keys keep this derivative away from sorting discontinuities.
    data = np.stack([rng.permutation(257) for _ in range(3)]).astype("float64")
    data = data / 8 + rng.uniform(-0.01, 0.01, size=data.shape)
    weights = np.linspace(-1.25, 2.75, data.size, dtype="float64").reshape(data.shape)
    x = jt.array(data, dtype="float64")
    p = jt.array(weights, dtype="float64")
    indices, values = jt.argsort(
        x, dim=1, descending=descending, dtype="int64", stable=True
    )
    loss = (values * p).sum()
    gradient = jt.grad(loss, x)
    assert str(indices.dtype) == "int64"
    for tensor in (values, loss, gradient):
        assert str(tensor.dtype) == "float64"
    _assert_resident(device, indices, values, loss, gradient)
    expected_indices = _stable_indices(data, 1, descending)
    expected_gradient = np.empty_like(data)
    np.put_along_axis(expected_gradient, expected_indices, weights, axis=1)
    np.testing.assert_array_equal(indices.numpy(), expected_indices)
    np.testing.assert_array_equal(
        values.numpy(), np.take_along_axis(data, expected_indices, axis=1)
    )
    np.testing.assert_allclose(
        gradient.numpy(), expected_gradient, rtol=1e-12, atol=1e-12
    )
