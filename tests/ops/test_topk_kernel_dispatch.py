"""Portable Topk keeps dimensions/options and the shared selection seam."""
import numpy as np
import pytest
import jittor as jt


@pytest.mark.parametrize("dim,k", [(0, 2), (1, 3), (-1, 3), (None, 3)])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("sorted", [False, True])
def test_cpu_topk_options_keep_portable_values_and_indices(dim, k, largest, sorted):
    source = np.array([[8., -1., 3., 11., 0.], [4., 13., 9., -2., 6.],
                       [7., 2., 14., 5., 12.]], dtype=np.float32)
    axis = -1 if dim is None else dim
    order = np.argsort(-source if largest else source, axis=axis)
    ids = np.take(order, np.arange(k), axis=axis)
    with jt.runtime.scope(use_cuda=0), jt.no_grad():
        values, indices = jt.topk(jt.array(source), k, dim, largest, sorted)
        np.testing.assert_array_equal(values.numpy(), np.take_along_axis(source, ids, axis))
        np.testing.assert_array_equal(indices.numpy(), ids)
        assert str(indices.dtype) == "int64"


def test_native_topk_passes_normalized_arguments_once_to_shared_selector(monkeypatch):
    from jittor.ops import sorting
    selected, calls = [], []
    result = (object(), object())
    def kernel(*args):
        calls.append(args)
        return result
    def select(name, *args):
        selected.append((name, args))
        return kernel
    monkeypatch.setattr(sorting, "select_kernel", select)
    with jt.runtime.scope(use_cuda=0):
        x = jt.ones((2, 17))
        assert sorting.topk(x, 3, -1, True, True) is result
    arguments = (x, 3, 1, True, True)
    assert selected == [("tensor.topk", arguments)] and calls == [arguments]
