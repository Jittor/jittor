"""Mapped inference concat and the retained training path on a real ACL device."""

import numpy as np
import pytest
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels.ops.concat_op import ConcatACL


@pytest.fixture(autouse=True)
def acl_inference():
    if not _test_capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required for mapped concat")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1), jt.no_grad(), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("dtype", ["float16", "float32", "int32", "int64", "bool"])
def test_native_provider_and_public_concat_match_numpy(dtype):
    left_np = (np.arange(6).reshape(2, 3) % 5).astype(dtype)
    right_np = (np.arange(10).reshape(2, 5) % 3).astype(dtype)
    left, right = jt.array(left_np, dtype=dtype), jt.array(right_np, dtype=dtype)
    expected = np.concatenate((left_np, right_np), axis=-1)
    for concat in (jt.core.ops.mapped_concat, ConcatACL(), jt.concat):
        output = concat([left, right], -1)
        assert output._producer_op() == "mapped_concat"
        assert not output._is_view()
        output.sync()
        assert output.location() == "device"
        assert str(output.dtype) == dtype
        np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("shapes,dim", [
    ([(0, 3), (2, 3)], 0),
    ([(2, 0), (2, 3), (2, 0)], -1),
    ([(0, 3), (0, 4)], 1),
    ([(2, 0), (3, 0)], 0),
    ([(0, 3)], 0),
])
def test_empty_extents_keep_numpy_shape_and_values(shapes, dim):
    arrays = [np.arange(np.prod(shape), dtype="float32").reshape(shape) for shape in shapes]
    output = ConcatACL()([jt.array(array) for array in arrays], dim)
    expected = np.concatenate(arrays, axis=dim)
    output.sync()
    assert tuple(output.shape) == expected.shape
    if expected.size:
        assert output.location() == "device"
    np.testing.assert_array_equal(output.numpy(), expected)


def test_single_input_is_an_independent_copy():
    values = np.arange(12, dtype="float32").reshape(3, 4)
    source = jt.array(values)
    result = ConcatACL()([source], -1)
    jt.sync([source, result])
    assert source._storage_address != result._storage_address
    assert not result._is_view()
    result.assign(result + 100)
    np.testing.assert_array_equal(source.numpy(), values)
    np.testing.assert_array_equal(result.numpy(), values + 100)


def test_dense_boundary_preserves_strided_and_broadcast_input_values():
    values = np.arange(24, dtype="float32").reshape(4, 6)
    base = jt.array(values)
    strided = jt.core.ops.getitem(base, (slice(None), slice(None, None, 2)))
    row = jt.array(np.array([[30, 31, 32]], dtype="float32"))
    broadcast = row.expand((4, 3))
    assert not strided._storage_is_contiguous()
    assert not broadcast._storage_is_contiguous()
    output = ConcatACL()([strided, broadcast], -1)
    output.sync()
    expected = np.concatenate((values[:, ::2], np.broadcast_to([[30, 31, 32]], (4, 3))), axis=-1)
    np.testing.assert_array_equal(output.numpy(), expected)


def test_input_count_and_axis_are_read_from_each_node():
    for count, dim in ((1, 0), (3, 1), (65, -1)):
        arrays = [np.full((2, 3), value, dtype="float32") for value in range(count)]
        output = ConcatACL()([jt.array(value) for value in arrays], dim)
        np.testing.assert_array_equal(output.numpy(), np.concatenate(arrays, axis=dim))


def test_native_rejects_cpu_training_and_invalid_metadata():
    with jt.flag_scope(use_cuda=0):
        cpu = jt.array(np.ones((2, 3), dtype="float32"))
        with pytest.raises(RuntimeError, match="ACL"):
            jt.core.ops.mapped_concat([cpu], 0)
    x = jt.ones((2, 3), dtype="float32")
    with jt.flag_scope(no_grad=0), pytest.raises(RuntimeError, match="no_grad"):
        jt.core.ops.mapped_concat([x], 0)
    cases = [
        ([], 0, "at least one"),
        ([x], 2, "dim out of range"),
        ([x, jt.ones((2, 3), dtype="float16")], 0, "same dtype"),
        ([x, jt.ones((2, 4), dtype="float32")], 0, "same shape"),
        ([x, jt.ones((2, 3, 1), dtype="float32")], 0, "same rank"),
    ]
    for inputs, dim, message in cases:
        with pytest.raises(RuntimeError, match=message):
            jt.core.ops.mapped_concat(inputs, dim)
        with pytest.raises(ValueError):
            ConcatACL()(inputs, dim)
    with pytest.raises(TypeError, match="list or tuple"):
        ConcatACL()(x, 0)
    with pytest.raises(TypeError, match="integer"):
        ConcatACL()([x], 0.5)


def test_training_keeps_codeop_and_split_gradients():
    left_np = np.arange(6, dtype="float32").reshape(2, 3) / 8
    right_np = np.arange(10, dtype="float32").reshape(2, 5) / 8
    with jt.flag_scope(no_grad=0):
        left, right = jt.array(left_np).start_grad(), jt.array(right_np).start_grad()
        output = ConcatACL()([left, right], -1)
        assert output._producer_op() == "code"
        gradients = jt.grad((output * output).sum(), [left, right])
        actual = jt.fetch_sync([output] + gradients)
    np.testing.assert_array_equal(actual[0], np.concatenate((left_np, right_np), axis=-1))
    np.testing.assert_array_equal(actual[1], 2 * left_np)
    np.testing.assert_array_equal(actual[2], 2 * right_np)
