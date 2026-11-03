"""Native HIP execution and library numerics, with CPU fallback forbidden."""

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.backend_libraries import get_library_ops
from jittor._runtime import forbid_backend_fallbacks


pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason="ROCm build required")


@pytest.fixture(autouse=True)
def native_rocm():
    with jt.runtime.scope(use_rocm=1, use_parallel_op_compiler=0):
        with forbid_backend_fallbacks():
            yield
            jt.sync_all(True)


def assert_device_value(value, expected, **tolerances):
    value.sync()
    assert value.location() == "device"
    assert value.device_id >= 0
    np.testing.assert_allclose(value.numpy(), expected, **tolerances)


def test_elementwise_reduction_and_gradient():
    x = jt.array([1.0, 2.0, 3.0])
    y = (x * 2 + 1).sqr().sum()
    gradient = jt.grad(y, x)
    assert_device_value(y, [83.0])
    assert_device_value(gradient, [12.0, 20.0, 28.0])


@pytest.mark.parametrize("shape,dim", [((1025,), 0), ((4, 129), 1), ((3, 9, 37), 2)])
def test_reductions_across_wavefronts(shape, dim):
    values = np.random.RandomState(7).normal(size=shape).astype("float32")
    x = jt.array(values)
    result = (x * 2).sum(dim)
    gradient = jt.grad(result.sum(), x)
    assert_device_value(result, (values * 2).sum(axis=dim), rtol=1e-5, atol=1e-5)
    assert_device_value(gradient, np.full_like(values, 2))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (True, False),
                                           (False, True), (True, True)])
def test_hipblas_transposes_and_gradients(dtype, trans_a, trans_b):
    a = (np.arange(12).reshape(3, 4) / 10).astype(dtype)
    b = (np.arange(20).reshape(4, 5) / 10).astype(dtype)
    a_input, b_input = (a.T.copy() if trans_a else a), (b.T.copy() if trans_b else b)
    x, y = jt.array(a_input).cast(dtype), jt.array(b_input).cast(dtype)
    product = get_library_ops("hipblas").hipblas_matmul(x, y, trans_a, trans_b)
    dx, dy = jt.grad(product.sum(), [x, y])
    da, db = np.ones((3, 5), dtype=dtype) @ b.T, a.T @ np.ones((3, 5), dtype=dtype)
    assert_device_value(product, a @ b, rtol=1e-5, atol=1e-6)
    assert_device_value(dx, da.T if trans_a else da, rtol=1e-5, atol=1e-6)
    assert_device_value(dy, db.T if trans_b else db, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("reverse", [False, True])
def test_rocprim_scan(dtype, reverse):
    values = (np.arange(18).reshape(3, 6) - 5).astype(dtype)
    x = jt.array(values).cast(dtype)
    result = get_library_ops("rocprim").rocprim_cumsum(x, reverse)
    expected = np.cumsum(values[:, ::-1] if reverse else values, axis=-1)
    if reverse:
        expected = expected[:, ::-1]
    assert_device_value(result, expected)
    if dtype.startswith("float"):
        gradient = jt.grad(result.sum(), x)
        expected_gradient = np.tile(np.arange(1, 7) if reverse else np.arange(6, 0, -1), (3, 1))
        assert_device_value(gradient, expected_gradient)


def test_public_matmul_training_steps_match_numpy():
    inputs = (np.arange(12).reshape(4, 3) / 12).astype("float32")
    target = (inputs @ np.array([[1.0], [-2.0], [3.0]], dtype="float32"))
    reference = np.zeros((3, 1), dtype="float32")
    x, y, weight = jt.array(inputs), jt.array(target), jt.array(reference.copy())
    for _ in range(5):
        prediction = jt.matmul(x, weight)
        loss = (prediction - y).sqr().mean()
        gradient = jt.grad(loss, weight)
        reference -= 0.1 * (2 / len(inputs)) * inputs.T @ (inputs @ reference - target)
        weight.assign(weight - 0.1 * gradient)
        assert_device_value(weight, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["uniform", "normal"])
@pytest.mark.parametrize("count", [1, 31, 4097])
def test_random_seed_replay_and_odd_lengths(dtype, kind, count):
    jt.set_seed(123)
    first = jt.random((count,), dtype=dtype, type=kind)
    second = jt.random((count,), dtype=dtype, type=kind)
    # Reseed before materializing: earlier lazy draws retain their own spans.
    jt.set_seed(123)
    replay = jt.random((count,), dtype=dtype, type=kind)
    replay_second = jt.random((count,), dtype=dtype, type=kind)
    first.sync()
    assert first.location() == "device"
    assert str(first.dtype) == dtype
    values = first.numpy()
    np.testing.assert_array_equal(replay.numpy(), values)
    np.testing.assert_array_equal(replay_second.numpy(), second.numpy())
    assert not np.array_equal(values, second.numpy())
    assert np.isfinite(values).all()
    if kind == "uniform":
        assert ((values >= 0) & (values <= 1)).all()
    if count == 4097:
        assert abs(values.mean() - (0.5 if kind == "uniform" else 0)) < 0.1
        assert abs(values.std() - (np.sqrt(1 / 12) if kind == "uniform" else 1)) < 0.1


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["uniform", "normal"])
def test_generator_state_and_split_draws(dtype, kind):
    generator = jt.Generator("cuda").manual_seed(987)
    state = generator.get_state()
    n, m = (5, 7) if dtype == "float32" and kind == "uniform" else (6, 8)
    first = jt.random((n,), dtype=dtype, type=kind, generator=generator)
    advanced = generator.get_state()
    second = jt.random((m,), dtype=dtype, type=kind, generator=generator)
    assert advanced != state
    generator.set_state(state)
    combined = jt.random((n + m,), dtype=dtype, type=kind, generator=generator)
    assert_device_value(combined, np.concatenate([first.numpy(), second.numpy()]), rtol=0, atol=0)
    generator.set_state(advanced)
    resumed = jt.random((m,), dtype=dtype, type=kind, generator=generator)
    assert_device_value(resumed, second.numpy(), rtol=0, atol=0)


def test_empty_random_and_linear_initialization():
    for kind in ("uniform", "normal"):
        for dtype in ("float32", "float64"):
            empty = jt.random((0,), dtype=dtype, type=kind)
            assert empty.numpy().shape == (0,)
    jt.set_seed(7)
    model = jt.nn.Sequential(jt.nn.Linear(3, 4), jt.nn.Relu(), jt.nn.Linear(4, 1))
    x = jt.array((np.arange(12).reshape(4, 3) / 12).astype("float32"))
    target = jt.array([[0.0], [0.5], [1.0], [1.5]])
    losses = []
    for _ in range(8):
        loss = (model(x) - target).sqr().mean()
        losses.append(float(loss.item()))
        parameters = model.parameters()
        for parameter, gradient in zip(parameters, jt.grad(loss, parameters)):
            parameter.assign(parameter - 0.1 * gradient)
        jt.sync_all(True)
        assert all(parameter.location() == "device" for parameter in parameters)
    assert losses[-1] < losses[0]
