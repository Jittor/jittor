"""Native batching must preserve logical scalar and outer-level boundaries."""

import numpy as np
import pytest
import jittor as jt

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if (
        request.param == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


def check(result, expected, device):
    assert tuple(result.shape) == np.shape(expected)
    result.sync()
    assert result.location() == ("device" if device == "cuda" else "cpu")
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", ["sum", "mean", "prod", "max", "min"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "bool"])
def test_logical_scalar_reduction(device, name, dtype):
    data = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    # Keep native dtype promotion, including bool reductions; identity alone
    # would preserve the wrong dtype even though the numerical value agrees.
    expected_dtype = str(getattr(jt.array(data[0, 0], dtype=dtype), name)().dtype)
    for dims in (None, [], ()):
        for keepdims in (False, True):
            x = jt.array(data, dtype=dtype)
            result = jt.vmap(
                jt.vmap(lambda value: getattr(value, name)(dims, keepdims))
            )(x)
            assert str(result.dtype) == expected_dtype
            if dtype.startswith("float"):
                gradient = jt.grad((result * result).sum(), x)
                second = jt.grad(gradient.sum(), x)
                # Materialize before numpy() changes any input residency.
                result.sync()
                gradient.sync()
                second.sync()
                check(gradient, 2 * data, device)
                check(second, np.full_like(data, 2), device)
            check(result, data, device)


@pytest.mark.parametrize("name", ["sum", "mean", "prod", "max", "min"])
def test_empty_axes_reduce_only_logical_axes(device, name):
    data = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    for dims in ([], ()):
        x = jt.array(data)
        result = jt.vmap(lambda row: getattr(row, name)(dims))(x)
        expected = getattr(np, name)(data, axis=(1, 2))
        check(result, expected, device)


@pytest.mark.parametrize("name", ["sum", "mean"])
def test_complex_logical_scalar_reduction(device, name):
    data = np.asarray([0.5 + 0.2j, -0.3 + 0.4j], dtype="complex64")
    x = jt.array(data)
    result = jt.vmap(lambda value: getattr(value, name)())(x)
    gradient = jt.grad((result * result.conj()).real.sum(), x)
    result.sync()
    gradient.sync()
    check(result, data, device)
    check(gradient, 2 * data, device)


@pytest.mark.parametrize(
    "name", ["sum", "mean", "prod", "max", "min", "argmax", "argmin"]
)
def test_invalid_axis_does_not_address_batch(device, name):
    x = jt.array(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    for axis in (-2, 1):
        with pytest.raises(ValueError, match="out of range"):
            jt.vmap(jt.vmap(lambda row: getattr(row, name)(axis)))(x)
    if name not in ("argmax", "argmin"):
        with pytest.raises(ValueError, match="out of range"):
            jt.vmap(jt.vmap(lambda row: getattr(row, name)([0, -2])))(x)


@pytest.mark.parametrize("nondefault_axes", [False, True])
def test_outer_shared_outputs_preserve_levels_and_gradients(device, nondefault_axes):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    weights = np.arange(8, dtype=np.float32).reshape(2, 4) / 4
    calls = []

    def outer(rows, shared):
        def inner(row, weight):
            calls.append(tuple(row.shape))
            return {"mapped": row + 1, "shared": (weight * weight, weight)}

        return jt.vmap(
            inner,
            in_dims=(-1 if nondefault_axes else 0, None),
            out_dims=-1 if nondefault_axes else 0,
        )(rows, shared)

    if nondefault_axes:
        data = data.transpose(2, 1, 0)
        weights = weights.T
    x, w = jt.array(data), jt.array(weights)
    result = jt.vmap(
        outer,
        in_dims=(2, 1) if nondefault_axes else (0, 0),
        out_dims=1 if nondefault_axes else 0,
    )(x, w)
    assert calls == [(4,)]
    expected = (
        np.repeat(weights[..., None], 3, axis=-1)
        if nondefault_axes
        else np.repeat(weights[:, None, :], 3, axis=1)
    )
    gradient = jt.grad(result["shared"][0].sum() + result["shared"][1].sum(), w)
    second = jt.grad(gradient.sum(), w)
    for value in (result["mapped"], *result["shared"], gradient, second):
        value.sync()
    check(result["shared"][0], expected**2, device)
    check(result["shared"][1], expected, device)
    check(
        result["mapped"],
        data.transpose(0, 2, 1) + 1 if nondefault_axes else data + 1,
        device,
    )
    check(gradient, 6 * weights + 3, device)
    check(second, np.full_like(weights, 6), device)


@pytest.mark.parametrize("shared_depth", [1, 2])
def test_three_levels_outer_shared_closure(device, shared_depth):
    x = jt.ones((2, 3, 5, 4))
    shape = (2, 4) if shared_depth == 1 else (2, 3, 4)
    weights = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    w = jt.array(weights)

    def outer(rows, shared):
        return jt.vmap(
            lambda middle, value: jt.vmap(lambda row: value)(middle),
            in_dims=(0, None if shared_depth == 1 else 0),
        )(rows, shared)

    result = jt.vmap(outer)(x, w)
    gradient = jt.grad(result.sum(), w)
    result.sync()
    gradient.sync()
    expanded = (
        weights[:, None, None, :] if shared_depth == 1 else weights[:, :, None, :]
    )
    check(result, np.broadcast_to(expanded, (2, 3, 5, 4)), device)
    check(gradient, np.full_like(weights, 15 if shared_depth == 1 else 5), device)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize(
    "source_shape,target_shape,dims,view_shape,output_shape",
    [
        ((4,), (3, 4), None, (1, 4), (3, 4)),
        ((4,), (3, 4), [], (1, 4), (3, 4)),
        ((4,), (3, 4), [-2], (1, 4), (3, 4)),
        ((1, 4), (3, 4), None, (1, 4), (3, 4)),
        ((2, 4), (4,), None, (2, 4), (2, 4)),
        ((4,), (2, 3, 4), [1], (1, 1, 4), (2, 3, 4)),
        ((4,), (2, 4, 3), [-1], (1, 4, 1), (2, 4, 3)),
        ((), (2, 3), None, (1, 1), (2, 3)),
        ((2, 4), (3,), [2], (2, 4, 1), (2, 4, 3)),
    ],
)
def test_broadcast_keeps_batch_axes(
    device, dtype, source_shape, target_shape, dims, view_shape, output_shape
):
    shape = (2, 3) + source_shape
    data = (np.arange(np.prod(shape)).reshape(shape) + 1).astype(dtype)
    if dtype == "complex64":
        data = data + 0.25j * data
    x = jt.array(data, dtype=dtype)
    result = jt.vmap(jt.vmap(lambda row: jt.broadcast(row, target_shape, dims)))(x)
    assert str(result.dtype) == dtype
    expected = np.broadcast_to(data.reshape((2, 3) + view_shape), (2, 3) + output_shape)
    factor = np.prod(output_shape) // np.prod(source_shape)
    loss = (result * result.conj()).real.sum()
    gradient = jt.grad(loss, x)
    second = jt.grad(gradient.sum(), x) if dtype == "float32" else None
    for value in (result, gradient, second):
        if value is not None:
            value.sync()
    check(result, expected, device)
    check(gradient, 2 * factor * data, device)
    if second is not None:
        check(second, np.full_like(data, 2 * factor), device)


def test_broadcast_nondefault_batch_axes_and_tensor_shape(device):
    data = np.arange(8, dtype=np.float64).reshape(4, 2)
    x = jt.array(data, dtype="float64")
    target = jt.zeros((3, 4), dtype="float64")
    result = jt.vmap(lambda row: row.broadcast(target), in_dims=1, out_dims=-1)(x)
    assert str(result.dtype) == "float64"
    gradient = jt.grad((result * result).sum(), x)
    result.sync()
    gradient.sync()
    check(result, np.broadcast_to(data, (3, 4, 2)), device)
    check(gradient, 6 * data, device)


def test_broadcast_rejects_invalid_logical_insert_axes(device):
    x = jt.ones((2, 4))
    for dims in ([2], [-3], [0, 0]):
        with pytest.raises(ValueError, match="out of range|duplicate"):
            jt.vmap(lambda row: row.broadcast((3, 4), dims))(x)


@pytest.mark.parametrize(
    "entry", ["method", "method_alias", "function", "function_alias"]
)
@pytest.mark.parametrize("shared_source", [False, True])
def test_broadcast_mapped_target_and_aliases(device, entry, shared_source):
    shape = (4,) if shared_source else (2, 4)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    x = jt.array(data)
    target = jt.zeros((2, 3, 4))
    calls = []

    def broadcast(source, reference):
        calls.append(tuple(reference.shape))
        if entry == "method":
            return source.broadcast(reference)
        if entry == "method_alias":
            return source.broadcast_var(reference)
        if entry == "function":
            return jt.broadcast(source, reference)
        return jt.broadcast_var(source, reference)

    result = jt.vmap(broadcast, in_dims=(None if shared_source else 0, 0))(x, target)
    assert calls == [(3, 4)]
    assert str(result.dtype) == "float32"
    source_gradient, target_gradient = jt.grad((result * result).sum(), [x, target])
    for value in (result, source_gradient, target_gradient):
        value.sync()
    expanded = data if shared_source else data[:, None, :]
    check(result, np.broadcast_to(expanded, (2, 3, 4)), device)
    check(source_gradient, (12 if shared_source else 6) * data, device)
    check(target_gradient, np.zeros((2, 3, 4)), device)
