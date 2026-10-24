"""Empty basic views and their setitem backward must not read a first slice."""
import numpy as np
import pytest

import jittor as jt
from _helpers import capability


@pytest.fixture(params=[0, 1], ids=["cpu", "cuda"])
def empty_slice_device(request):
    if request.param and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("CUDA unavailable")
    with jt.flag_scope(use_cuda=request.param, backend_fallback="error"):
        before = jt.core.backend_fallback_count()
        yield "cuda" if request.param else "cpu"
        jt.sync_all(True)
        assert jt.core.backend_fallback_count() == before


def assert_resident(value, device):
    value.sync()
    assert jt.core.dispatch_context([value])[0] == device


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
def test_empty_tuple_forward_and_complete_gradient(empty_slice_device, shape):
    raw = np.arange(np.prod(shape), dtype="float32").reshape(shape) + 1
    x = jt.array(raw)
    # The direct operator preserves the empty VarSlices form that the
    # transpose-view source also uses; the Python indexing path is separate.
    y = jt.core.ops.getitem(x, ())
    gradient = jt.grad((y * y).sum(), x)
    assert_resident(y, empty_slice_device)
    assert_resident(gradient, empty_slice_device)
    assert y._storage_address == x._storage_address
    assert y._storage_offset() == x._storage_offset()
    assert tuple(y._storage_strides()) == tuple(x._storage_strides())
    np.testing.assert_array_equal(y.numpy(), raw)
    np.testing.assert_array_equal(gradient.numpy(), 2 * raw)


def test_transpose_matmul_complete_gradients(empty_slice_device):
    left = np.arange(12, dtype="float32").reshape(3, 4) + 1
    right = np.arange(6, dtype="float32").reshape(3, 2) / 4 + 1
    weight = np.arange(8, dtype="float32").reshape(4, 2) / 8 + 1
    a, b = jt.array(left), jt.array(right)
    result = jt.matmul(a.transpose(), b)
    ga, gb = jt.grad((result * jt.array(weight)).sum(), [a, b])
    for value in (result, ga, gb):
        assert_resident(value, empty_slice_device)
    np.testing.assert_allclose(result.numpy(), left.T @ right, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ga.numpy(), right @ weight.T, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gb.numpy(), left @ weight, rtol=1e-6, atol=1e-6)


def test_empty_tuple_write_preserves_both_complete_gradients(empty_slice_device):
    raw = np.arange(6, dtype="float32").reshape(2, 3) + 1
    x, v = jt.array(raw), jt.array(raw / 2)
    result = jt.core.ops.setitem(x, (), v * 3)
    gx, gv = jt.grad((result * result).sum(), [x, v])
    for value in (result, gx, gv):
        assert_resident(value, empty_slice_device)
    np.testing.assert_array_equal(result.numpy(), raw * 1.5)
    np.testing.assert_array_equal(gx.numpy(), np.zeros_like(raw))
    np.testing.assert_array_equal(gv.numpy(), raw * 9)
