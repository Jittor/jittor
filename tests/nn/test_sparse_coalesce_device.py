"""COO coalesce runs on the operands' device (jittor-core-gaps.md §3.8).

coalesce() used to pull row, col and values to NumPy for np.unique/np.add.at
even under CUDA. The dedup now sorts, prefix-sums and reduces on the device;
only the merged count (the output shape) is read back. Every case is checked
against that NumPy algorithm, which remains the reference here.
"""

import numpy as np
import pytest
import jittor as jt
from jittor import sparse

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


def _reference(row, col, values, n_cols):
    key = row.astype(np.int64) * n_cols + col.astype(np.int64)
    unique, inverse = np.unique(key, return_inverse=True)
    merged = np.zeros((len(unique),) + values.shape[1:], dtype=values.dtype)
    np.add.at(merged, inverse.reshape(-1), values)
    return unique // n_cols, unique % n_cols, merged, inverse.reshape(-1)


def _coo(row, col, values, shape, index_dtype):
    indices = jt.array(np.stack([row, col]).astype(index_dtype), dtype=index_dtype)
    return sparse.sparse_array(indices, jt.array(values, dtype=str(values.dtype)),
                               jt.NanoVector(list(shape)))


@pytest.mark.parametrize("shape, nnz, trailing", [
    ((7, 5), 40, ()),            # heavy duplication
    ((3, 1000), 12, ()),         # wide, few duplicates
    ((1, 1), 9, ()),             # every entry the same coordinate
    ((6, 4), 30, (3,)),          # values with a trailing dimension
    ((2 ** 20, 2 ** 20), 64, ()),  # large shape: must not densify
])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_matches_numpy_reference(device, shape, nnz, trailing, index_dtype):
    rng = np.random.default_rng(nnz + shape[1])
    row = rng.integers(0, shape[0], nnz)
    col = rng.integers(0, shape[1], nnz)
    values = rng.standard_normal((nnz,) + trailing)
    result = _coo(row, col, values, shape, index_dtype).coalesce()
    result.indices.sync(device_sync=True, weak_sync=False)
    result.values.sync(device_sync=True, weak_sync=False)
    expected_location = "device" if device == "cuda" else "cpu"
    assert result.indices.location() == expected_location
    assert result.values.location() == expected_location
    assert str(result.indices.dtype) == index_dtype
    ref_row, ref_col, ref_values, _ = _reference(row, col, values, shape[1])
    np.testing.assert_array_equal(result.indices.numpy(), np.stack([ref_row, ref_col]))
    np.testing.assert_allclose(result.values.numpy(), ref_values, rtol=1e-12, atol=1e-12)


def test_integer_values_sum_exactly(device):
    row = np.array([0, 1, 0, 2, 1])
    col = np.array([1, 0, 1, 2, 0])
    values = np.array([2 ** 40, 3, 5, 7, -3], dtype=np.int64)
    result = _coo(row, col, values, (3, 3), "int64").coalesce()
    np.testing.assert_array_equal(result.values.numpy(), [2 ** 40 + 5, 0, 7])
    np.testing.assert_array_equal(result.indices.numpy(), [[0, 1, 2], [1, 0, 2]])


def test_zero_nnz(device):
    result = _coo(np.zeros(0, int), np.zeros(0, int), np.zeros(0), (4, 3), "int64").coalesce()
    assert tuple(result.indices.shape) == (2, 0)
    assert tuple(result.values.shape) == (0,)
    np.testing.assert_array_equal(result.to_dense().numpy(), np.zeros((4, 3)))


def test_value_gradient_gathers_the_merged_gradient(device):
    row = np.array([2, 0, 2, 1, 0, 2])
    col = np.array([1, 3, 1, 0, 3, 1])
    values_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    indices = jt.array(np.stack([row, col]), dtype="int64")
    values = jt.array(values_np, dtype="float64")
    result = sparse.sparse_array(indices, values, jt.NanoVector([3, 4])).coalesce()
    weights = np.array([10.0, 20.0, 30.0])
    grad = jt.grad((result.values * jt.array(weights, dtype="float64")).sum(), values).numpy()
    _, _, _, inverse = _reference(row, col, values_np, 4)
    np.testing.assert_allclose(grad, weights[inverse])
