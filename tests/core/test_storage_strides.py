"""Materialized stride-zero views remain usable by generated kernels."""

from _helpers import capability as _test_capability
import numpy as np
import pytest
import jittor as jt

pytestmark = pytest.mark.usefixtures("storage_device")


@pytest.fixture(params=("cpu", "cuda"))
def storage_device(request):
    if request.param == "cuda" and not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA compiler and device required")
    with jt.flag_scope(use_cuda=int(request.param == "cuda"), backend_fallback="error"):
        yield


def test_expanded_storage_consumers_and_readback():
    a = jt.array([[1.], [2.]])
    a.sync()
    b = a.expand(2, 3)
    b.sync()
    assert b._storage_address == a._storage_address
    assert b.raw_ptr == a.raw_ptr
    assert list(b._storage_strides()) == [1, 0]
    expected = np.array([[1., 1., 1.], [2., 2., 2.]])
    np.testing.assert_array_equal(b.numpy(), expected)
    assert b.data.strides == (4, 0)
    assert not b.data.flags.writeable
    added = b + b
    added.sync()
    if jt.introspection.policy.runtime.use_cuda:
        assert added.location() == "device"
    np.testing.assert_array_equal(added.numpy(), expected * 2)
    np.testing.assert_array_equal((-b).numpy(), -expected)
    np.testing.assert_array_equal(b.sum(1).numpy(), [3, 6])
    np.testing.assert_array_equal(jt.where(b > 1, b, b + 2).numpy(), [[3, 3, 3], [2, 2, 2]])
    np.testing.assert_array_equal(b.reindex([3, 2], ["i1", "i0"]).numpy(), expected.T)
    np.testing.assert_array_equal(b.reindex_reduce("add", [2], ["i0"]).numpy(), [3, 6])
    np.testing.assert_array_equal(b.copy().numpy(), expected)
    dense = b.contiguous()
    dense.sync()
    assert dense.raw_ptr != b.raw_ptr
    np.testing.assert_array_equal(dense.numpy(), expected)


def test_expand_and_materialized_slice_follow_source_assignment():
    a = jt.array([[1.], [2.]])
    b = a.expand(2, 3)
    v = b[1:2]
    b.sync()
    v.sync()
    a.assign(jt.array([[4.], [5.]]))
    np.testing.assert_array_equal(b.numpy(), [[4, 4, 4], [5, 5, 5]])
    np.testing.assert_array_equal(v.numpy(), [[5, 5, 5]])
    with pytest.raises(RuntimeError, match="overlapping storage"):
        b.assign(jt.ones((2, 3)))


def test_expanded_view_and_explicit_dense_library_boundary():
    a = jt.array([[1.], [2.]])
    b = a.expand(2, 3)
    v = b.view(2, 1, 3)
    v.sync()
    assert v.raw_ptr == a.raw_ptr
    with pytest.raises(RuntimeError, match="incompatible with storage strides"):
        b.view(6)
    np.testing.assert_array_equal(b.reshape(6).numpy(), [1, 1, 1, 2, 2, 2])
    # Opaque code kernels declare dense input access; the graph gets an
    # explicit contiguous node instead of reinterpreting physical storage.
    out = jt.code(b.shape, b.dtype, [b],
                  cpu_src="for (int i=0;i<6;++i) out0_p[i]=in0_p[i]*3;",
                  cuda_src="""
__global__ void scale3(@ARGS_DEF) {
    @PRECALC
    int i=threadIdx.x;
    if (i<6) out0_p[i]=in0_p[i]*3;
}
scale3<<<1,32>>>(@ARGS);
""")
    np.testing.assert_array_equal(out.numpy(), [[3, 3, 3], [6, 6, 6]])


def test_expanded_storage_gradient_reduces_broadcast_axes():
    a = jt.array([[1.], [2.]])
    b = a.expand(2, 3)
    b.sync()
    gradient = jt.grad((b*b).sum(), a)
    np.testing.assert_array_equal(gradient.numpy(), [[6], [12]])


def test_positive_step_slice_has_storage_offset_and_strides():
    a = jt.array(np.arange(8, dtype=np.float32))
    v = a[1::2]
    assert v._storage_offset() == 1
    assert list(v._storage_strides()) == [2]
    v.sync()
    assert v._storage_address == a._storage_address + 4
    np.testing.assert_array_equal(v.numpy(), [1, 3, 5, 7])
    np.testing.assert_array_equal((v*2).numpy(), [2, 6, 10, 14])


def test_strided_integer_indices_are_dense_at_native_indexing_boundary():
    index_base = np.array([[1, 5, 6, 7, 8], [1, 9, 10, 11, 12]], dtype=np.int64)
    indices = jt.array(index_base)[:, 4:]
    assert not indices._storage_is_contiguous()

    weight_data = np.arange(128 * 3, dtype=np.float32).reshape(128, 3)
    gathered = jt.array(weight_data)[indices]
    expected = weight_data[index_base[:, 4:]]
    np.testing.assert_array_equal(gathered.numpy(), expected)

    target = jt.zeros((16,), dtype="float32")
    target[indices] = jt.array([[2.0], [3.0]], dtype="float32")
    expected_target = np.zeros((16,), dtype=np.float32)
    expected_target[index_base[:, 4:]] = [[2.0], [3.0]]
    np.testing.assert_array_equal(target.numpy(), expected_target)

    source_data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = jt.array(source_data)[:, :, ::2]
    source_index = jt.array([1], dtype="int64")
    assert not source._storage_is_contiguous()
    np.testing.assert_array_equal(
        source[source_index].numpy(),
        source_data[:, :, ::2][[1]],
    )


def test_single_integer_array_index_on_inner_axis():
    source_data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    expected = source_data[..., [-2], :]

    for index in ([-2], jt.array([-2], dtype="int64")):
        source = jt.array(source_data)
        first = source[..., index, :].numpy()
        second = source[..., index, :].numpy()
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)


def test_split_along_inner_dimension_preserves_outer_rows():
    source_data = np.arange(2 * 8 * 48, dtype=np.float32).reshape(2, 8, 48)
    chunks = jt.array(source_data).split(16, dim=2)

    assert len(chunks) == 3
    for index, chunk in enumerate(chunks):
        np.testing.assert_array_equal(
            chunk.numpy(), source_data[:, :, index * 16:(index + 1) * 16]
        )


def test_split_transpose_views_are_dense_at_batched_matmul_boundary():
    rng = np.random.default_rng(20260908)
    x_data = rng.standard_normal((2, 8, 16), dtype=np.float32)
    weight_data = rng.standard_normal((16, 48), dtype=np.float32)

    projected = np.matmul(x_data, weight_data)
    query_data, key_data, _ = np.split(projected, 3, axis=2)
    query_data = query_data.reshape(2, 8, 2, 8).transpose(0, 2, 1, 3)
    key_data = key_data.reshape(2, 8, 2, 8).transpose(0, 2, 1, 3)
    expected = np.matmul(query_data, key_data.swapaxes(-1, -2))

    def run_once():
        projected = jt.matmul(jt.array(x_data), jt.array(weight_data))
        query, key, _ = projected.split(16, dim=2)
        query = query.view(2, 8, 2, 8).transpose(1, 2)
        key = key.view(2, 8, 2, 8).transpose(1, 2)
        return jt.matmul(query, key.transpose(-1, -2)).numpy()

    first = run_once()
    second = run_once()
    np.testing.assert_allclose(first, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(second, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(first, second)
