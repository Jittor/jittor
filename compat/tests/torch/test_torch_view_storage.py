"""A view has to share the storage it was taken from.

`y = x.transpose(0, 1)` is not a new tensor in torch: it is the same storage
described differently, so `y.stride()` reflects the swap, `y.is_contiguous()` is
false, and `y[0, 1] = 99` writes through to `x`. A shim that materialises the
view instead gets the *shape* right and nothing else, and the difference only
shows up where it matters -- an optimizer writing through a slice, tied weights
that are supposed to be one buffer, safetensors asked to refuse a
non-contiguous tensor.

These are the reproductions from the "非连续 Tensor 与 Safetensors 兼容问题"
section of the C5 issue doc. The doc filed it as C5 on the grounds that no
native pointer+stride+offset constructor exists; on this branch one does, and
is in production use -- `getitem_op.cc` already emits a strided output sharing
its input's allocation, which is why a slice writes through today.
"""
import contextlib

import jittor as jt
import numpy as np
import torch


@contextlib.contextmanager
def storage_view_transpose():
    """Return a permutation as a view of its input, the way torch does.

    Off by default: it changes what every consumer downstream of a transpose
    receives. They handle it -- elementwise, reduce, broadcast and reindex read
    `storage_strides`, `reinterpret_view` and `code` `USER_CHECK` contiguity,
    and a slice already reaches `.numpy()`, `.cpu()` and the fused kernels as a
    strided Var today -- but a cuBLAS or cuTT path that wants dense memory may
    still prefer one copy up front. That is a benchmark, not a correctness
    question, and it has not been run.
    """
    previous = jt.flags.transpose_storage_view
    jt.flags.transpose_storage_view = 1
    try:
        yield
    finally:
        jt.flags.transpose_storage_view = previous


# -- what holds with the default, materialising transpose ---------------------

def test_transpose_writes_through_to_its_base():
    x = torch.arange(6).reshape(2, 3).float()
    y = x.transpose(0, 1)
    y[0, 1] = 99.0
    assert x.numpy().reshape(-1).tolist()[3] == 99.0, x.numpy()


def test_a_plain_tensor_is_contiguous():
    x = torch.arange(6).reshape(2, 3).float()
    assert x._storage_is_contiguous()
    assert tuple(x._storage_strides()) == (3, 1)


def test_as_strided_shares_storage():
    a = torch.arange(6).float()
    v = a.as_strided((2, 2), (1, 2))
    np.testing.assert_array_equal(v.numpy().reshape(-1), [0, 2, 1, 3])
    v[0, 0] = 77.0
    assert a.numpy()[0] == 77.0, a.numpy()


# -- and what the native storage-view transpose adds --------------------------

def test_as_strided_reports_its_own_stride():
    # The values and the write-through above hold either way; the *stride
    # report* does not, because the composition that serves a permuted
    # `as_strided` ends in a transpose. With the default one, the buffer really
    # is dense by then and `(2, 1)` is the true answer about it.
    with storage_view_transpose():
        a = torch.arange(6).float()
        v = a.as_strided((2, 2), (1, 2))
        assert tuple(v._storage_strides()) == (1, 2), tuple(v._storage_strides())
        assert not v._storage_is_contiguous()


def test_storage_view_transpose_reports_the_swapped_stride():
    with storage_view_transpose():
        x = torch.arange(6).reshape(2, 3).float()
        y = x.transpose(0, 1)
        assert tuple(y.shape) == (3, 2)
        assert tuple(y._storage_strides()) == (1, 3), tuple(y._storage_strides())
        assert not y._storage_is_contiguous()


def test_storage_view_transpose_still_computes_the_right_numbers():
    # The point of the strides is that a kernel reads through them. A view that
    # reports (1, 3) and then hands the next op a dense read is worse than no
    # view at all, so check the values and a reduction over them.
    with storage_view_transpose():
        x = torch.arange(6).reshape(2, 3).float()
        y = x.transpose(0, 1)
        np.testing.assert_array_equal(y.numpy(), [[0, 3], [1, 4], [2, 5]])
        assert float(y.sum()) == 15.0


def test_storage_view_transpose_writes_through():
    with storage_view_transpose():
        x = torch.arange(6).reshape(2, 3).float()
        y = x.transpose(0, 1)
        y[0, 1] = 99.0
        assert x.numpy().reshape(-1).tolist() == [0, 1, 2, 99, 4, 5], x.numpy()


def test_the_flag_is_put_back():
    before = jt.flags.transpose_storage_view
    with storage_view_transpose():
        pass
    assert jt.flags.transpose_storage_view == before
