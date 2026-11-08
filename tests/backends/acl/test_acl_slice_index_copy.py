"""310P positive-slice inference assignments preserve IndexPut semantics."""

import numpy as np
import pytest
import jittor as jt

from _helpers import capability as _test_capability
from jittor._core.dtypes import dtype_name


def require_310p():
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip('ACL backend is unavailable')
    from jittor.backends.acl.kernels._soc import soc_name
    if not soc_name().startswith('Ascend310P'):
        pytest.skip('The StridedSliceAssignV2 replacement targets 310P')


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('rhs_dtype', ['float16', 'float32'])
@pytest.mark.parametrize('axis', [0, 1, 2, 3])
@pytest.mark.parametrize('selection', [slice(0, None, 2), slice(1, None, 2), slice(0, None, 3), slice(1, -1, 1)])
def test_positive_slice_copy_preserves_inputs(dtype, rhs_dtype, axis, selection, monkeypatch):
    require_310p()
    from jittor.backends.acl.kernels.ops import setitem_op
    shape = (3, 5, 7, 8)
    slices = [slice(None)] * len(shape)
    slices[axis] = selection
    slices = tuple(slices)
    source_shape = list(shape)
    source_shape[axis] = len(range(*selection.indices(shape[axis])))
    original = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 8).astype(dtype)
    values = (-np.arange(np.prod(source_shape), dtype=np.float32).reshape(source_shape) / 7 - 1).astype(rhs_dtype)
    expected = original.copy()
    expected[slices] = values.astype(dtype)
    routed = []
    old_call = setitem_op.IndexCopyACL.__call__
    def record(self, base, dim, index, source):
        routed.append((dim, tuple(index.shape), dtype_name(index.dtype)))
        return old_call(self, base, dim, index, source)
    monkeypatch.setattr(setitem_op.IndexCopyACL, '__call__', record)
    with jt.runtime.scope(use_cuda=1), jt.no_grad():
        base, source = jt.array(original), jt.array(values.astype(dtype))
        result = setitem_op.SetItemACL()(base, slices, source)
        jt.sync([result, base, source])
        assert result is not base
        assert result._storage_address != base._storage_address
        assert result._storage_address != source._storage_address
        assert dtype_name(result.dtype) == dtype
        assert all(v.location() == 'device' for v in (result, base, source))
        actual, unchanged_base, unchanged_source = jt.fetch_sync([result, base, source])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(unchanged_base, original)
        np.testing.assert_array_equal(unchanged_source, values.astype(dtype))
        # Public assignment casts the source before backend dispatch and updates
        # its destination holder. Other aliases are not used as the oracle.
        public_base, public_source = jt.array(original), jt.array(values)
        public_index = (Ellipsis, selection) if axis == len(shape)-1 else slices
        public_base[public_index] = public_source
        jt.sync([public_base, public_source])
        assert dtype_name(public_base.dtype) == dtype
        assert all(v.location() == 'device' for v in (public_base, public_source))
        actual, unchanged_source = jt.fetch_sync([public_base, public_source])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(unchanged_source, values)
    assert routed == [(axis, (source_shape[axis],), 'int32')] * 2


@pytest.mark.parametrize('case', ['broadcast', 'multiple_axes', 'integer', 'negative_step',
                                  'empty', 'full_slice', 'dtype_mismatch', 'bool',
                                  'strided_base', 'strided_source', 'other_soc'])
def test_positive_slice_copy_declines_other_forms(case, monkeypatch):
    require_310p()
    from jittor.backends.acl.kernels.ops import setitem_op
    def unexpected(*args, **kwargs):
        pytest.fail('Assignment outside the positive-slice inference subset reached IndexCopy')
    monkeypatch.setattr(setitem_op.IndexCopyACL, '__call__', unexpected)
    with jt.runtime.scope(use_cuda=1, transpose_storage_view=1), jt.no_grad():
        base = jt.array(np.arange(24, dtype=np.float32).reshape(4, 6))
        value = jt.ones((4, 3), dtype='float32')
        slices = (slice(None), slice(None, None, 2))
        if case == 'broadcast': value = jt.ones((1, 3), dtype='float32')
        elif case == 'multiple_axes':
            slices = (slice(None, None, 2), slice(None, None, 2))
            value = jt.ones((2, 3), dtype='float32')
        elif case == 'integer': slices, value = (1, slice(None)), jt.ones((6,), dtype='float32')
        elif case == 'negative_step': slices, value = (slice(None), slice(None, None, -1)), jt.ones((4, 6), dtype='float32')
        elif case == 'empty': slices, value = (slice(None), slice(6, None, 2)), jt.empty((4, 0), dtype='float32')
        elif case == 'full_slice': slices, value = (slice(None), slice(None)), jt.ones((4, 6), dtype='float32')
        elif case == 'dtype_mismatch': value = value.cast('float16')
        elif case == 'bool': base, value = base.cast('bool'), value.cast('bool')
        elif case == 'strided_base':
            base = jt.array(np.arange(24, dtype=np.float32).reshape(6, 4)).transpose(1, 0)
            assert not base._storage_is_contiguous()
        elif case == 'strided_source':
            value = jt.ones((3, 4), dtype='float32').transpose(1, 0)
            assert not value._storage_is_contiguous()
        elif case == 'other_soc': monkeypatch.setattr(setitem_op, 'soc_name', lambda: 'Ascend910B')
        assert setitem_op._positive_slice_index_copy(base, slices, value) is None


def test_positive_slice_copy_keeps_training_gradient_path(monkeypatch):
    require_310p()
    from jittor.backends.acl.kernels.ops import setitem_op
    def unexpected(*args, **kwargs):
        pytest.fail('Training must retain the differentiable IndexPut path')
    monkeypatch.setattr(setitem_op.IndexCopyACL, '__call__', unexpected)
    with jt.runtime.scope(use_cuda=1), jt.enable_grad():
        base = jt.array(np.arange(18, dtype=np.float32).reshape(3, 6))
        value = jt.array(-np.arange(9, dtype=np.float32).reshape(3, 3))
        slices = (slice(None), slice(None, None, 2))
        assert setitem_op._positive_slice_index_copy(base, slices, value) is None
        result = setitem_op.SetItemACL()(base, slices, value)
        grad_base, grad_value = jt.grad(result.sum(), [base, value])
        jt.sync([result, grad_base, grad_value])
        assert all(v.location() == 'device' for v in (result, grad_base, grad_value))
        actual_base, actual_value = jt.fetch_sync([grad_base, grad_value])
        expected_base = np.ones((3, 6), dtype=np.float32)
        expected_base[:, ::2] = 0
        np.testing.assert_array_equal(actual_base, expected_base)
        np.testing.assert_array_equal(actual_value, np.ones((3, 3), dtype=np.float32))
