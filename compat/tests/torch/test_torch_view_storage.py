"""A view has to share the storage it was taken from.

`y = x.transpose(0, 1)` is not a new tensor in torch: it is the same storage
described differently, so `y.stride()` reflects the swap, `y.is_contiguous()` is
false, and `y[0, 1] = 99` writes through to `x`. A shim that materialises the
view instead gets the *shape* right and nothing else, and the difference only
shows up in the places that matter -- an optimizer writing through a slice,
tied weights that are supposed to be one buffer, safetensors asked to refuse a
non-contiguous tensor.

These are the reproductions from the "非连续 Tensor 与 Safetensors 兼容问题"
section of the C5 issue doc. The doc filed it as C5 -- no native constructor --
but this branch's `VarHolder` exposes `_set_storage_view_of`,
`_set_transpose_view_of`, `_storage_strides` and `_storage_offset` to Python,
so the exit condition it names is met and the gap is in compat.
"""
import numpy as np
import pytest
import torch


#: Why the two stride-reporting tests below are expected to fail.
#:
#: This branch's views are *logical*: `transpose` and `as_strided` materialise
#: dense data and record how to re-derive it, which is what makes writing
#: through them reach the base. `_storage_strides()` then reports what the
#: buffer actually is -- dense -- and that report is true. Torch's contract is
#: the other construction: one buffer, described by permuted strides, nothing
#: copied.
#:
#: Closing the gap means making these produce a Var that shares its base's
#: allocation with non-dense `storage_strides`, and that is only safe once every
#: consumer either honours strides or refuses. Elementwise, reduce, broadcast
#: and reindex already read them, and `reinterpret_view`/`code` already
#: `USER_CHECK(is_contiguous())`, so the invariant is maintained deliberately --
#: but the sweep has not been done, and a consumer that reads the buffer densely
#: would be silently wrong rather than loud.
#:
#: Making `_storage_strides()` report the logical strides *without* that change
#: would be the worse half: a claim about a physical layout that is not there.
_PHYSICAL_VIEW = pytest.mark.xfail(
    reason="views here are logical (materialised data + recorded derivation); "
           "physical strided views need a stride-honouring sweep of every "
           "consumer first",
    strict=True,
)


@_PHYSICAL_VIEW
def test_transpose_reports_the_swapped_stride():
    x = torch.arange(6).reshape(2, 3).float()
    y = x.transpose(0, 1)
    assert tuple(y.shape) == (3, 2)
    assert tuple(y._storage_strides()) == (1, 3), tuple(y._storage_strides())
    assert not y._storage_is_contiguous()


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


@_PHYSICAL_VIEW
def test_as_strided_reports_its_own_stride():
    a = torch.arange(6).float()
    v = a.as_strided((2, 2), (1, 2))
    assert tuple(v._storage_strides()) == (1, 2), tuple(v._storage_strides())
    assert not v._storage_is_contiguous()
