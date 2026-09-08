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
