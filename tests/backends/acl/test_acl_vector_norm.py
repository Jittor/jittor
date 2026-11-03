"""Measured 310P last-axis vector norm, safe half accumulation and admission."""
import numpy as np
import pytest
import jittor as jt

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels import normalization
from jittor.backends.acl.kernels._soc import soc_name


@pytest.fixture(autouse=True)
def acl_device():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    if not soc_name().startswith("Ascend310P"):
        pytest.skip("vector norm inference is validated on Ascend310P only")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, transpose_storage_view=1), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("shape", [(2, 5, 17), (17,)])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("scale", [1., 1000., 1e-4])
def test_public_last_axis_norm_preserves_dtype_and_safe_accumulation(shape, dtype, keepdim, scale, monkeypatch):
    from jittor.backends.acl.kernels.ops import vector_norm_op
    observed = []
    original = vector_norm_op.VectorNormACL.__call__
    def observe(self, x, keep):
        observed.append((tuple(x.shape), keep))
        return original(self, x, keep)
    monkeypatch.setattr(vector_norm_op.VectorNormACL, "__call__", observe)
    with jt.no_grad():
        for slot in range(2):
            source = (np.random.RandomState(20261030 + slot).normal(size=shape) * scale).astype(dtype)
            x = jt.array(source)
            result = jt.linalg.vector_norm(x, ord=2, dim=(-1,), keepdim=keepdim)
            result.sync()
            assert result.location() == "device"
            assert str(result.dtype) == dtype
            expected = np.linalg.norm(source.astype(np.float64), axis=-1, keepdims=keepdim).astype(dtype)
            actual = result.numpy()
            assert tuple(actual.shape) == tuple(expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=1e-3 if dtype == "float16" else 2e-5, atol=1e-7)
            np.testing.assert_array_equal(x.numpy(), source)
    assert observed == [(shape, keepdim)] * 2


def test_strided_normalize_includes_copy_and_retains_epsilon_and_inputs(monkeypatch):
    from jittor.backends.acl.kernels.ops import vector_norm_op
    original = vector_norm_op.VectorNormACL.__call__
    seen = []
    def observe(self, x, keep):
        assert x._storage_is_contiguous()
        seen.append(keep)
        return original(self, x, keep)
    monkeypatch.setattr(vector_norm_op.VectorNormACL, "__call__", observe)
    with jt.no_grad():
        for dtype in ("float16", "float32"):
            for scale in (0., 1e-5, 1.):
                source = (np.random.RandomState(31).normal(size=(2, 5, 3, 17)) * scale).astype(dtype)
                x = jt.array(source).transpose(0, 2, 1, 3)
                assert not x._storage_is_contiguous()
                result = jt.nn.normalize(x, p=2, dim=-1, eps=1e-4)
                result.sync()
                assert result.location() == "device" and str(result.dtype) == dtype
                logical = source.transpose(0, 2, 1, 3)
                denominator = np.linalg.norm(logical.astype(np.float64), axis=-1, keepdims=True).astype(dtype)
                expected = logical / np.maximum(denominator, np.asarray(1e-4, dtype=dtype))
                np.testing.assert_allclose(result.numpy(), expected, rtol=1e-3 if dtype == "float16" else 2e-5, atol=1e-7)
                np.testing.assert_array_equal(x.numpy(), logical)
    assert seen == [True] * 6


def test_declines_training_other_soc_multi_axis_and_unverified_metadata(monkeypatch):
    supports = normalization.vector_norm_supported
    x = jt.ones((2, 5, 17)).float16()
    with jt.no_grad():
        assert supports(x, 2, -1, True)
        assert supports(x, 2, (2,), False)
        for dim in (None, (), (1, 2), (2, 2), 0, 3, -4, True):
            assert not supports(x, 2, dim, True)
        for order in (0, 1, 3, "fro", float("inf")):
            assert not supports(x, order, -1, True)
        assert not supports(x.int32(), 2, -1, True)
        assert not supports(x.float64(), 2, -1, True)
        assert not supports(jt.empty((2, 0)).float16(), 2, -1, True)
        for soc in ("Ascend910B3", "Ascend910", ""):
            monkeypatch.setattr(normalization, "soc_name", lambda: soc)
            assert not supports(x, 2, -1, True)
    monkeypatch.setattr(normalization, "soc_name", lambda: "Ascend310P3")
    with jt.enable_grad():
        assert not supports(x, 2, -1, True)
        source = np.asarray([[1., 2., 3.], [-2., 3., 4.]], dtype="float32")
        value = jt.array(source)
        norm = jt.linalg.vector_norm(value, ord=2, dim=-1)
        grad = jt.grad(norm.sum(), value)
        np.testing.assert_allclose(grad.numpy(), source / np.linalg.norm(source, axis=-1, keepdims=True), rtol=2e-5, atol=1e-6)


def test_direct_owner_refuses_gradient_mode():
    from jittor.backends.acl.kernels.ops.vector_norm_op import VectorNormACL
    with jt.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        VectorNormACL()(jt.ones((2, 3)))


def test_float32_vector_norm_special_values_and_default_zero_normalize():
    with jt.no_grad():
        source = np.asarray([[np.nan, 1., 2.], [np.inf, 1., 2.], [-np.inf, 1., 2.], [0., 0., 0.]], dtype="float32")
        result = jt.linalg.vector_norm(jt.array(source), ord=2, dim=-1, keepdim=True)
        actual = result.numpy()
        assert np.isnan(actual[0, 0])
        assert np.isposinf(actual[1:3, 0]).all()
        assert actual[3, 0] == 0
        np.testing.assert_array_equal(jt.nn.normalize(jt.zeros((2, 3)), dim=-1).numpy(), np.zeros((2, 3), dtype="float32"))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_normalize_epsilon_denominator_boundary(dtype):
    source = np.asarray([[5e-5, 0.], [1e-4, 0.], [2e-4, 0.]], dtype=dtype)
    expected = source / np.maximum(np.linalg.norm(source.astype("float64"), axis=-1, keepdims=True).astype(dtype), np.asarray(1e-4, dtype=dtype))
    with jt.no_grad():
        result = jt.nn.normalize(jt.array(source), p=2, dim=-1, eps=1e-4)
        assert str(result.dtype) == dtype
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-3 if dtype == "float16" else 2e-5, atol=1e-7)
