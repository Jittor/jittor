"""Native strided-batch hipBLAS values, gradients and public dispatch on a DCU."""

import numpy as np
import pytest

import jittor as jt
from jittor._runtime import forbid_backend_fallbacks
from jittor._runtime.backend_libraries import get_library_ops
from jittor._runtime.dispatch import select_kernel


pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason="ROCm build required")


@pytest.fixture(autouse=True)
def native_rocm():
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_rocm=1, use_parallel_op_compiler=0):
        with forbid_backend_fallbacks():
            yield
            jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before


def assert_device(value, expected, dtype):
    value.sync()
    if np.size(expected):
        assert value.location() == "device"
        assert value.device_id >= 0
    assert str(value.dtype) == dtype
    tolerance = 2e-5 if dtype == "float32" else 1e-11
    np.testing.assert_allclose(value.numpy(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("batch", [(3,), (2, 3)])
@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (True, False),
                                           (False, True), (True, True)])
def test_native_transposes_and_nonuniform_gradients(dtype, batch, trans_a, trans_b):
    # Binary fractions keep the fixture exact even under auto_convert_64.
    a = (np.arange(np.prod(batch) * 12).reshape(batch + (3, 4)) % 17 - 8) / 8
    b = (np.arange(np.prod(batch) * 20).reshape(batch + (4, 5)) % 13 - 6) / 8
    g = (np.arange(np.prod(batch) * 15).reshape(batch + (3, 5)) % 11 - 5) / 16
    a, b, g = (v.astype(dtype) for v in (a, b, g))
    raw_a = a.swapaxes(-1, -2).copy() if trans_a else a
    raw_b = b.swapaxes(-1, -2).copy() if trans_b else b
    x, y = jt.array(raw_a).cast(dtype), jt.array(raw_b).cast(dtype)
    output = get_library_ops("hipblas").hipblas_batched_matmul(x, y, trans_a, trans_b)
    dx, dy = jt.grad((output * jt.array(g).cast(dtype)).sum(), [x, y])
    da = g @ b.swapaxes(-1, -2)
    db = a.swapaxes(-1, -2) @ g
    assert_device(output, a @ b, dtype)
    assert_device(dx, da.swapaxes(-1, -2) if trans_a else da, dtype)
    assert_device(dy, db.swapaxes(-1, -2) if trans_b else db, dtype)


@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (True, False),
                                                   (False, True), (True, True)])
def test_public_dispatch_with_batch_permutation_and_transpose_views(transpose_a, transpose_b):
    rng = np.random.RandomState(41)
    a = rng.normal(size=(3, 2, 3, 4)).astype("float32")
    b = rng.normal(size=(3, 2, 4, 5)).astype("float32")
    raw_a = a.swapaxes(-1, -2).copy() if transpose_a else a
    raw_b = b.swapaxes(-1, -2).copy() if transpose_b else b
    x0, y0 = jt.array(raw_a), jt.array(raw_b)
    x, y = x0.transpose((1, 0, 2, 3)), y0.transpose((1, 0, 2, 3))
    if transpose_a:
        x = x.transpose((0, 1, 3, 2))
    if transpose_b:
        y = y.transpose((0, 1, 3, 2))
    assert select_kernel("batched_matmul", x, y, False, False).__name__ == "_hipblas_batched_matmul"
    result = jt.matmul(x, y)
    g = rng.normal(size=(2, 3, 3, 5)).astype("float32")
    dx, dy = jt.grad((result * jt.array(g)).sum(), [x0, y0])
    actual_a, actual_b = a.transpose(1, 0, 2, 3), b.transpose(1, 0, 2, 3)
    da = (g @ actual_b.swapaxes(-1, -2)).transpose(1, 0, 2, 3)
    db = (actual_a.swapaxes(-1, -2) @ g).transpose(1, 0, 2, 3)
    assert_device(result, actual_a @ actual_b, "float32")
    assert_device(dx, da.swapaxes(-1, -2) if transpose_a else da, "float32")
    assert_device(dy, db.swapaxes(-1, -2) if transpose_b else db, "float32")
    np.testing.assert_array_equal(x0.numpy(), raw_a)
    np.testing.assert_array_equal(y0.numpy(), raw_b)


def test_batch_broadcast_keeps_the_existing_same_device_path():
    a = (np.arange(24).reshape(2, 1, 3, 4) / 8).astype("float32")
    b = (np.arange(60).reshape(1, 3, 4, 5) / 16).astype("float32")
    x, y = jt.array(a), jt.array(b)
    assert select_kernel("batched_matmul", x, y, False, False) is None
    result = jt.matmul(x, y)
    assert_device(result, a @ b, "float32")


@pytest.mark.parametrize("shape_a,shape_b", [((0, 3, 4), (0, 4, 5)),
                                          ((2, 0, 4), (2, 4, 5)),
                                          ((2, 3, 0), (2, 0, 5))])
def test_empty_batch_rows_and_inner_dimension(shape_a, shape_b):
    a, b = np.zeros(shape_a, dtype="float32"), np.zeros(shape_b, dtype="float32")
    assert_device(jt.matmul(jt.array(a), jt.array(b)), a @ b, "float32")


@pytest.mark.parametrize("length", [128, 512])
def test_attention_products_dispatch_to_hipblas(length):
    rng = np.random.RandomState(59)
    q = rng.normal(size=(1, 4, length, 16)).astype("float32")
    k = rng.normal(size=q.shape).astype("float32")
    v = (rng.normal(size=q.shape) / np.sqrt(length)).astype("float32")
    x, y, z = (jt.array(value) for value in (q, k, v))
    scores = jt.matmul(x, y.transpose((0, 1, 3, 2)))
    assert_device(scores, q @ k.swapaxes(-1, -2), "float32")
    output = jt.matmul(scores, z)
    assert_device(output, (q @ k.swapaxes(-1, -2)) @ v, "float32")


@pytest.mark.parametrize("policy", ["prefer16", "prefer_bfloat16"])
def test_half_amp_does_not_select_a_float32_output_provider(policy):
    x = jt.array(np.ones((2, 3, 4), dtype="float32"))
    y = jt.array(np.ones((2, 4, 5), dtype="float32"))
    with jt.flag_scope(amp_reg=getattr(jt.amp_flags, policy)):
        assert select_kernel("batched_matmul", x, y, False, False) is None


def test_shared_operand_accumulates_both_gradient_contributions():
    values = (np.arange(24).reshape(2, 3, 4) / 8).astype("float32")
    seed = (np.arange(18).reshape(2, 3, 3) / 16).astype("float32")
    x = jt.array(values)
    output = jt.matmul(x, x.transpose((0, 2, 1)))
    gradient = jt.grad((output * jt.array(seed)).sum(), x)
    expected = seed @ values + seed.swapaxes(-1, -2) @ values
    assert_device(gradient, expected, "float32")


def test_batched_gradient_retains_its_derivative_graph():
    a = (np.arange(24).reshape(2, 3, 4) / 8).astype("float32")
    b = (np.arange(40).reshape(2, 4, 5) / 16).astype("float32")
    seed = (np.arange(30).reshape(2, 3, 5) / 32).astype("float32")
    x, y = jt.array(a), jt.array(b)
    output = jt.matmul(x, y)
    first = jt.grad((output * jt.array(seed)).sum(), x)
    second = jt.grad(first.sum(), y)
    expected = np.broadcast_to(seed.sum(axis=1, keepdims=True), b.shape)
    assert_device(second, expected, "float32")


def test_float64_is_not_admitted_by_the_fp32_provider():
    x = jt.array64(np.ones((2, 3, 4)))
    y = jt.array64(np.ones((2, 4, 5)))
    assert select_kernel("batched_matmul", x, y, False, False) is None
    with pytest.raises(jt.JittorError, match="matching float32"):
        get_library_ops("hipblas").hipblas_batched_matmul(x, y, False, False)
