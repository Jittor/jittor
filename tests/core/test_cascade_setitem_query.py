"""Backend routing must not feed custom results to native view writeback."""

import jittor as jt
import numpy as np
import pytest


def test_cascade_query_matches_integer_ancestry_without_gradient_dependency():
    x = jt.array(np.arange(12, dtype="float32").reshape(3, 4)).stop_grad()
    assert not x._needs_cascade_setitem()
    row = jt.core.ops.getitem(x, 1)
    assert row._needs_cascade_setitem()
    assert not jt.core.ops.getitem(x, slice(None))._needs_cascade_setitem()
    assert not jt.core.ops.getitem(x, (1, slice(None)))._needs_cascade_setitem()
    assert not (row + 1)._needs_cascade_setitem()


def test_native_cascade_still_updates_held_parent():
    x = jt.array(np.arange(12, dtype="float32").reshape(3, 4)).stop_grad()
    row = jt.core.ops.getitem(x, 1)
    updated = jt.core.ops.setitem(row, 2, jt.array(90.0))
    row.check_cascade_setitem(updated)
    expected = np.arange(12, dtype="float32").reshape(3, 4)
    expected[1, 2] = 90
    np.testing.assert_array_equal(x.numpy(), expected)


def test_cascade_rejects_a_non_setitem_result_before_native_downcast():
    x = jt.ones((3, 4))
    row = jt.core.ops.getitem(x, 1)
    with pytest.raises(RuntimeError, match="requires a native setitem result"):
        row.check_cascade_setitem(row + 1)
    np.testing.assert_array_equal(x.numpy(), np.ones((3, 4)))


@pytest.mark.parametrize("use_cuda", [0, 1])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_basic_indexing_gradients_preserve_low_precision_assignments(use_cuda, dtype):
    if use_cuda and not jt.has_cuda:
        pytest.skip("CUDA is unavailable")
    with jt.flag_scope(use_cuda=use_cuda, backend_fallback="error"):
        x = jt.array(np.arange(8, dtype="float32").reshape(2, 4)).cast(dtype)
        weights = jt.array(np.array([0.5, 1.5, 2.5, 3.5], dtype="float32"))
        row = jt.core.ops.getitem(x, 1)
        gradient = jt.grad((row * weights).sum(), x)
        expected = np.zeros((2, 4), dtype="float32")
        expected[1] = [0.5, 1.5, 2.5, 3.5]
        assert str(gradient.dtype) == dtype
        np.testing.assert_array_equal(gradient.float32().numpy(), expected)
        value = jt.ones((4,), dtype=dtype)
        result = jt.core.ops.setitem(x, 1, value)
        gradient = jt.grad(result.sum(), x)
        expected[0], expected[1] = 1, 0
        assert str(gradient.dtype) == dtype
        np.testing.assert_array_equal(gradient.float32().numpy(), expected)
