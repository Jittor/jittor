"""Actual 310P Topk inference, legal selection and narrow admission."""
import numpy as np
import pytest
import jittor as jt

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels import tensor
from jittor.backends.acl.kernels._soc import soc_name


@pytest.fixture(autouse=True)
def acl_device():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    if not soc_name().startswith("Ascend310P"):
        pytest.skip("Topk inference admission is validated on Ascend310P only")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, transpose_storage_view=1), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("rows,width,k", [(2, 17, 1), (3, 128, 17), (1, 128, 128), (1, 8400, 300)])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("pattern", ["unique", "ties", "equal"])
@pytest.mark.parametrize("dim", [-1, 1])
def test_public_topk_values_indices_and_alternating_inputs(rows, width, k, dtype, pattern, dim, monkeypatch):
    from jittor.backends.acl.kernels.ops import topk_op
    seen = []
    original = topk_op.TopkACL.__call__
    def observe(self, value, count):
        seen.append((tuple(value.shape), count))
        return original(self, value, count)
    monkeypatch.setattr(topk_op.TopkACL, "__call__", observe)
    with jt.no_grad():
        for slot in range(2):
            rng = np.random.RandomState(20261029 + slot)
            if pattern == "unique":
                pool = (np.arange(width, dtype=np.uint16) + 0x0800).view(np.float16)
                source = np.stack([pool[rng.permutation(width)] for _ in range(rows)]).astype(dtype)
            elif pattern == "ties":
                source = rng.randint(-8, 9, (rows, width)).astype(dtype)
            else:
                source = np.full((rows, width), -3.982421875 + slot, dtype=dtype)
            values, indices = jt.topk(jt.array(source), k, dim=dim, largest=True, sorted=True)
            jt.sync([values, indices], True)
            assert values.location() == indices.location() == "device"
            assert str(values.dtype) == dtype and str(indices.dtype) == "int64"
            actual, ids = values.numpy(), indices.numpy()
            assert actual.shape == ids.shape == (rows, k)
            assert np.all(ids >= 0) and np.all(ids < width)
            assert all(len(np.unique(row)) == k for row in ids)
            np.testing.assert_array_equal(actual, np.take_along_axis(source, ids, -1))
            np.testing.assert_array_equal(actual, np.sort(source, axis=-1)[:, ::-1][:, :k])
            if pattern == "unique":
                np.testing.assert_array_equal(ids, np.argsort(-source, axis=-1)[:, :k])
    assert seen == [((rows, width), k)] * 2


def test_topk_declines_other_soc_grad_layout_and_unverified_arguments(monkeypatch):
    x = jt.ones((2, 17)).float16()
    supports = tensor.topk_supported
    with jt.no_grad():
        assert supports(x, 3)
        for name in ("Ascend910B3", "Ascend910", ""):
            monkeypatch.setattr(tensor, "soc_name", lambda: name)
            assert not supports(x, 3)
        monkeypatch.setattr(tensor, "soc_name", lambda: "Ascend310P3")
        for k, dim, largest, sorted in ((0, -1, True, True), (18, -1, True, True),
                (True, -1, True, True), (3, 0, True, True), (3, -1, False, True),
                (3, -1, True, False)):
            assert not supports(x, k, dim, largest, sorted)
        assert not supports(x.int32(), 3)
        assert not supports(jt.ones((1, 2, 17)).float16(), 3)
        view = jt.core.ops.transpose(jt.ones((17, 2)).float16(), (1, 0))
        assert not view._storage_is_contiguous()
        assert not supports(view, 3)
    with jt.enable_grad():
        assert not supports(x, 3)


def test_direct_topk_inference_owner_rejects_training():
    from jittor.backends.acl.kernels.ops.topk_op import TopkACL
    with jt.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        TopkACL()(jt.ones((2, 17)), 3)
