"""310P inference gather, legal indices, independent outputs and decline scope."""
import numpy as np
import pytest

import jittor as jt

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels.ops import getitem_op
from jittor.backends.acl.kernels._soc import soc_name


@pytest.fixture(autouse=True)
def acl_device():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    if not soc_name().startswith("Ascend310P"):
        pytest.skip("single-axis inference admission is validated on Ascend310P only")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, transpose_storage_view=1), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


CASES = (
    ((225, 3), 0, 4096, False, False),
    ((225, 6), 0, 4096, False, False),
    ((225, 12), 0, 4096, False, False),
    ((225, 24), 0, 4096, False, False),
    ((2, 17, 5), 1, 11, False, False),
    ((2, 3, 19), 2, 13, False, False),
    ((23, 7), 0, 17, False, False),
    ((2, 17, 5), 1, 11, True, False),
    ((3, 19, 5), 1, 13, False, True),
    ((19,), 0, 11, False, False),
    ((2, 3, 17, 5), 2, 11, False, False),
)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_public_single_axis_legal_values_and_alternating_inputs(case, dtype, index_dtype, monkeypatch):
    shape, axis, count, source_strided, index_strided = case
    rng = np.random.RandomState(20261029)
    calls = []
    original = getitem_op.GatherV2ACL.__call__

    def record(self, source, dim, index):
        calls.append((dim, tuple(index.shape)))
        return original(self, source, dim, index)

    monkeypatch.setattr(getitem_op.GatherV2ACL, "__call__", record)
    with jt.no_grad():
        for slot in range(2):
            values = rng.uniform(-1, 1, shape).astype(dtype)
            indices = rng.randint(0, shape[axis], count).astype(index_dtype)
            indices[::2] -= shape[axis]
            indices[:3] = [-1, -shape[axis], -1]
            x = jt.array(values)
            if source_strided:
                x = jt.core.ops.transpose(jt.array(values.swapaxes(-2, -1).copy()), (0, 2, 1))
                assert not x._storage_is_contiguous()
            index = jt.array(indices)
            if index_strided:
                backing = np.full(count * 2, 100003, dtype=index_dtype)
                backing[::2] = indices
                index = jt.array(backing)[::2]
                assert not index._storage_is_contiguous()
            slices = [slice(None)] * len(shape)
            slices[axis] = index
            result = x[tuple(slices)]
            result.sync(True)
            assert result.location() == "device" and str(result.dtype) == dtype
            assert result._storage_address != x._storage_address
            expected = np.take(values, indices, axis=axis)
            np.testing.assert_array_equal(result.numpy(), expected)
    assert calls == [(axis, (count,)), (axis, (count,))]


@pytest.mark.parametrize("form", ["grad", "other_soc", "bool", "empty", "multiple", "newaxis", "partial", "list", "host_index"])
def test_unverified_forms_keep_the_original_path(form, monkeypatch):
    x = jt.ones((4, 5)).float16()
    index = jt.array([3, 0, -1], dtype="int64")
    with jt.no_grad():
        slices = index
        if form == "other_soc":
            monkeypatch.setattr(getitem_op, "soc_name", lambda: "Ascend910B3")
        elif form == "bool":
            slices = jt.array([True, False, True, False])
        elif form == "empty":
            slices = jt.array(np.empty((0,), dtype=np.int64))
        elif form == "multiple":
            slices = (index, jt.array([1, 2, 3], dtype="int64"))
        elif form == "newaxis":
            slices = (None, index)
        elif form == "partial":
            slices = (index, slice(1, None))
        elif form == "list":
            slices = [3, 0, -1]
        elif form == "host_index":
            token = jt.core._set_construction_placement(0, 0)
            try:
                slices = jt.array([3, 0, -1], dtype="int64")
            finally:
                jt.core._restore_construction_placement(token)
            assert slices.placement_backend == 0
        if form == "grad":
            with jt.enable_grad():
                assert getitem_op._single_axis_gather_spec(x, slices) is None
        else:
            assert getitem_op._single_axis_gather_spec(x, slices) is None


def test_newaxis_keeps_old_execution_and_ellipsis_uses_gather(monkeypatch):
    values = np.arange(20, dtype=np.float32).reshape(4, 5) / 8
    calls = []
    original = getitem_op.GatherV2ACL.__call__

    def record(self, source, dim, index):
        calls.append(dim)
        return original(self, source, dim, index)

    monkeypatch.setattr(getitem_op.GatherV2ACL, "__call__", record)
    with jt.no_grad():
        x, index = jt.array(values), jt.array([3, 0, -1], dtype="int32")
        expanded = x[None, index]
        expanded.sync(True)
        np.testing.assert_array_equal(expanded.numpy(), values[None, [3, 0, -1]])
        assert calls == []
        selected = x[index, ...]
        selected.sync(True)
        np.testing.assert_array_equal(selected.numpy(), values[[3, 0, -1], ...])
        assert calls == [0]


def test_direct_primitive_rejects_grad_mode():
    with jt.enable_grad():
        with pytest.raises(RuntimeError, match="requires no_grad"):
            getitem_op.GatherV2ACL()(jt.ones((4, 5)).float16(), 0,
                                     jt.array([1, 0], dtype="int32"))
