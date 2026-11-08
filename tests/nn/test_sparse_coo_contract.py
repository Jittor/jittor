"""COO construction, metadata, and sparse numerical contracts."""

import numpy as np
import pytest
import jittor as jt

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        capability.require_accelerator("cuda")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


def make_coo(indices, values, shape):
    return jt.sparse.sparse_array(indices, values, jt.NanoVector(shape))


@pytest.mark.parametrize("index_dtype", ["int32", "int64", "uint8"])
@pytest.mark.parametrize("value_dtype", ["float32", "float64", "complex64"])
def test_metadata_preserves_input_identity_and_dtype(device, index_dtype, value_dtype):
    indices = jt.array([[0, 1, 1], [2, 0, 2]], dtype=index_dtype)
    values = jt.array([1, 2, 3], dtype=value_dtype)
    sparse = make_coo(indices, values, [2, 3])
    assert sparse.nnz == 3
    assert sparse.dtype == values.dtype
    assert str(sparse.dtype) == value_dtype
    assert sparse.is_sparse() is True
    assert sparse.ndim == 2
    assert tuple(sparse.shape) == (2, 3)
    assert sparse._indices() is indices
    assert sparse._values() is values


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("nnz", [0, 3])
def test_constructor_keeps_ndimensional_and_zero_nnz_storage(device, ndim, nnz):
    indices = jt.array(np.zeros((ndim, nnz), dtype=np.int32))
    values = jt.array(np.arange(nnz, dtype=np.float32))
    sparse = make_coo(indices, values, [2] * ndim)
    assert sparse.nnz == nnz
    assert sparse.ndim == ndim
    assert sparse._indices() is indices
    assert sparse._values() is values


def test_constructor_keeps_values_with_trailing_dimensions(device):
    # The existing storage contract only requires a leading nnz dimension;
    # this does not claim that matrix-only sparse operations accept it.
    indices = jt.array([[0, 1, 1], [2, 0, 2]], dtype="int32")
    values = jt.array(np.arange(6, dtype=np.float32).reshape(3, 2))
    sparse = make_coo(indices, values, [2, 3])
    assert sparse.nnz == 3
    assert sparse._values() is values


@pytest.mark.parametrize("index_shape", [(), (3,), (2, 1, 3)])
def test_constructor_rejects_indices_rank(device, index_shape):
    indices = jt.array(np.zeros(index_shape, dtype=np.int32))
    values = jt.array([1, 2, 3], dtype="float32")
    # Stop at construction: malformed coordinates must never reach a kernel.
    with pytest.raises(ValueError, match="indices.*shape"):
        make_coo(indices, values, [2, 3])


@pytest.mark.parametrize("coordinate_dims", [1, 3])
def test_constructor_rejects_coordinate_dimension_mismatch(device, coordinate_dims):
    indices = jt.array(np.zeros((coordinate_dims, 3), dtype=np.int32))
    values = jt.array([1, 2, 3], dtype="float32")
    with pytest.raises(ValueError, match="indices.*shape"):
        make_coo(indices, values, [2, 3])


@pytest.mark.parametrize("value_shape", [(), (0,), (2,), (4,)])
def test_constructor_rejects_values_nnz_mismatch(device, value_shape):
    indices = jt.array([[0, 1, 1], [2, 0, 2]], dtype="int32")
    values = jt.array(np.zeros(value_shape, dtype=np.float32))
    with pytest.raises(ValueError, match="values.*nnz"):
        make_coo(indices, values, [2, 3])


@pytest.mark.parametrize("index_dtype", ["bool", "float32", "float64", "complex64"])
def test_constructor_rejects_noninteger_indices(device, index_dtype):
    indices = jt.array([[0, 1, 1], [2, 0, 2]], dtype=index_dtype)
    values = jt.array([1, 2, 3], dtype="float32")
    with pytest.raises(TypeError, match="indices.*integer"):
        make_coo(indices, values, [2, 3])


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_valid_coo_roundtrip_and_value_gradient_execute_on_device(device, index_dtype):
    indices = jt.array([[0, 1, 1], [2, 0, 2]], dtype=index_dtype)
    values = jt.array([1, 2, 3], dtype="float32")
    sparse = make_coo(indices, values, [2, 3])
    dense = sparse.to_dense()
    gradient = jt.grad(dense.sum(), values)
    roundtrip = sparse.to_csr().to_coo()
    restored = roundtrip.to_dense()
    assert roundtrip.nnz == 3
    assert roundtrip.dtype == values.dtype
    assert str(roundtrip._indices().dtype) == index_dtype
    assert roundtrip.is_sparse() is True
    # numpy() fetches device storage to CPU; check placement before any fetch.
    results = (dense, gradient, restored)
    for result in results:
        result.sync()
    for result in results:
        assert result.location() == ("device" if device == "cuda" else "cpu")
    expected = np.array([[0, 0, 1], [2, 0, 3]], dtype=np.float32)
    np.testing.assert_array_equal(dense.numpy(), expected)
    np.testing.assert_array_equal(restored.numpy(), expected)
    np.testing.assert_array_equal(gradient.numpy(), np.ones(3, dtype=np.float32))


def test_coalesce_preserves_float64_values_and_gradient(device):
    indices = jt.array([[1, 0, 1], [2, 1, 2]], dtype="int64")
    # These binary fractions are exact in float64 but disappear in float32.
    source = np.array([1 + 2**-40, -3 + 2**-39, 2 + 2**-38], dtype=np.float64)
    values = jt.array(source, dtype="float64")
    sparse = make_coo(indices, values, [2, 3])
    with jt.flag_scope(auto_convert_64_to_32=1):
        result = sparse.coalesce()
    weights_np = np.array([2 + 2**-42, -1 + 2**-41], dtype=np.float64)
    weights = jt.array(weights_np, dtype="float64")
    gradient = jt.grad((result._values() * weights).sum(), values)
    assert result.nnz == 2
    assert str(result.dtype) == "float64"
    assert str(result._indices().dtype) == "int64"
    assert str(gradient.dtype) == "float64"
    # Coalesce still computes on the host. This checks output placement, not
    # a claim that the deduplication itself has become a device-side kernel.
    for tensor in (result._indices(), result._values(), gradient):
        tensor.sync()
        assert tensor.location() == ("device" if device == "cuda" else "cpu")
    np.testing.assert_array_equal(result._indices().numpy(), [[0, 1], [1, 2]])
    np.testing.assert_array_equal(
        result._values().numpy(), np.array([source[1], source[0] + source[2]])
    )
    np.testing.assert_array_equal(gradient.numpy(), weights_np[[1, 0, 1]])


def test_coalesce_preserves_int64_values(device):
    indices = jt.array([[1, 0, 1], [2, 1, 2]], dtype="int64")
    source = np.array([2**40 + 3, -(2**40) + 7, 2**40 + 5], dtype=np.int64)
    values = jt.array(source, dtype="int64")
    sparse = make_coo(indices, values, [2, 3])
    with jt.flag_scope(auto_convert_64_to_32=1):
        result = sparse.coalesce()
    assert result.nnz == 2
    assert str(result.dtype) == "int64"
    result._values().sync()
    assert result._values().location() == ("device" if device == "cuda" else "cpu")
    np.testing.assert_array_equal(
        result._values().numpy(), np.array([source[1], source[0] + source[2]])
    )


@pytest.mark.parametrize("large_axis", [0, 1], ids=["large_row", "large_column"])
def test_coalesce_preserves_large_int64_coordinates_without_densifying(
    device, monkeypatch, large_axis
):
    large = 2**31 + 17
    if large_axis == 0:
        coordinates = [[large, 1, large], [2, 0, 2]]
        shape = [large + 2, 3]
        expected = [[1, large], [0, 2]]
    else:
        coordinates = [[2, 0, 2], [large, 1, large]]
        shape = [3, large + 2]
        expected = [[0, 2], [1, large]]
    indices = jt.array(coordinates, dtype="int64")
    values = jt.array([1.25, 2.0, 3.75], dtype="float64")
    sparse = make_coo(indices, values, shape)

    def forbidden_dense(_self):
        raise AssertionError("large sparse coordinates must never be densified")

    monkeypatch.setattr(jt.sparse.SparseVar, "to_dense", forbidden_dense)
    # Only three valid nonzeros are allocated; even row * ncols + col fits
    # int64. Do not convert this large logical shape to CSR or a dense tensor.
    with jt.flag_scope(auto_convert_64_to_32=1):
        result = sparse.coalesce()
    assert tuple(result.shape) == tuple(shape)
    assert result.nnz == 2
    assert str(result._indices().dtype) == "int64"
    for tensor in (result._indices(), result._values()):
        tensor.sync()
        assert tensor.location() == ("device" if device == "cuda" else "cpu")
    np.testing.assert_array_equal(result._indices().numpy(), expected)
    np.testing.assert_array_equal(result._values().numpy(), [2.0, 5.0])
