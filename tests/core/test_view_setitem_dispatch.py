"""Explicit view writeback accepts backend results without producer inspection."""

import jittor as jt
import numpy as np
import pytest


def test_tensor_indexing_records_views_but_operator_entry_does_not():
    x = jt.array(np.arange(12, dtype="float32").reshape(3, 4)).stop_grad()
    assert not x._is_view()
    assert x[1]._is_view()
    assert x[1:3]._is_view()
    row = jt.core.ops.getitem(x, 1)
    assert not row._is_view()
    assert not (row + 1)._is_view()


def test_recorded_view_accepts_a_non_setitem_result():
    x = jt.array(np.arange(12, dtype="float32").reshape(3, 4)).stop_grad()
    row = x[1]
    row.assign(row + 90)
    expected = np.arange(12, dtype="float32").reshape(3, 4)
    expected[1] += 90
    np.testing.assert_array_equal(x.numpy(), expected)


def test_backend_updated_value_writes_through_recorded_slice(monkeypatch):
    from jittor.ops import indexing
    x = jt.zeros((5, 4))
    view = x[1:4]
    seen = []

    def dispatch(name, value, *args):
        if name == "tensor.setitem":
            seen.append(value._is_view())
            return jt.ones(value.shape) * 7
        return None

    monkeypatch.setattr(indexing, "try_dispatch", dispatch)
    view[1] = 99
    assert seen == [True]
    expected = np.zeros((5, 4), dtype="float32")
    expected[1:4] = 7
    np.testing.assert_array_equal(x.numpy(), expected)


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
