"""Stock RMSNorm uses ACL inference without changing precision or boundaries."""
import numpy as np
import pytest
import torch
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks


def _require_acl():
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip('real ACL device required')


def _values(rows, width, slot=0, dtype='float16'):
    rng = np.random.default_rng(20261022 + rows * 100 + width + slot)
    source = rng.normal(0, 1.25, (rows, width)).astype(dtype)
    if slot:
        source[0] *= np.asarray(.0001, dtype=dtype)
    weight = rng.normal(.9, .3, (width,)).astype(dtype)
    return source, weight


def _reference(source, weight, eps, axes=(-1,)):
    wide = source.astype(np.float64)
    return (wide / np.sqrt((wide * wide).mean(axes, keepdims=True) + eps)
            * weight.astype(np.float64)).astype(source.dtype)


def _module(weight, shape=None, eps=1e-6, dtype=torch.float16):
    shape = tuple(weight.shape) if shape is None else shape
    module = torch.nn.RMSNorm(shape, eps=eps, dtype=dtype, device='cuda')
    module.weight = torch.nn.Parameter(torch.tensor(weight, dtype=dtype, device='cuda'))
    return module.eval()


def _record_dispatch(monkeypatch):
    from jittor.compat.torch.installers.nn import extra_api
    original = extra_api._try_dispatch
    calls = []
    def record(operation, *args, **kwargs):
        calls.append(operation)
        return original(operation, *args, **kwargs)
    monkeypatch.setattr(extra_api, '_try_dispatch', record)
    return calls


@pytest.mark.parametrize('width', [16, 17, 31, 32, 33, 64, 128, 3072, 4096])
@pytest.mark.parametrize('rows', [1, 2, 17])
def test_stock_half_rms_norm_acl_matches_wide_reference(width, rows, monkeypatch):
    _require_acl()
    calls = _record_dispatch(monkeypatch)
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, auto_graph_replay=0), torch.no_grad(), forbid_backend_fallbacks():
        for slot in range(2):
            source, weight = _values(rows, width, slot)
            value = torch.tensor(source, dtype=torch.float16, device='cuda')
            module = _module(weight)
            result = module(value)
            jt.sync_all(True)
            assert result.location() == 'device' and result.dtype == torch.float16
            actual = result.detach().cpu().numpy()
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, _reference(source, weight, module.eps), rtol=3e-3, atol=3e-3)
            np.testing.assert_array_equal(value.detach().cpu().numpy(), source)
            np.testing.assert_array_equal(module.weight.detach().cpu().numpy(), weight)
    assert calls == ['nn.rms_norm.inference'] * 2
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize('width', [1, 2, 7, 8, 15])
def test_310p_small_half_rms_norm_declines_corrupt_vendor_kernel(width, monkeypatch):
    _require_acl()
    from jittor.backends.acl.kernels._soc import soc_name
    if not soc_name().startswith('Ascend310P'):
        pytest.skip('small-width vendor issue was reproduced on 310P')
    from jittor.backends.acl.kernels.normalization import _rms_norm_cuda_acl
    calls = _record_dispatch(monkeypatch)
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, auto_graph_replay=0), torch.no_grad(), forbid_backend_fallbacks():
        source, weight = _values(17, width)
        value = torch.tensor(source, dtype=torch.float16, device='cuda')
        module = _module(weight)
        assert _rms_norm_cuda_acl(value, module.weight, module.eps) is None
        result = module(value)
        jt.sync_all(True)
        assert result.location() == 'device' and result.dtype == torch.float16
        np.testing.assert_allclose(result.cpu().numpy(), _reference(source, weight, module.eps), rtol=3e-3, atol=3e-3)
    assert calls == []
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize('width', [1, 2, 7, 8, 15])
@pytest.mark.parametrize('route', ['mapped', 'code'])
def test_310p_direct_small_width_vendor_routes_fail_closed(width, route):
    _require_acl()
    from jittor.backends.acl.kernels._soc import soc_name
    if not soc_name().startswith('Ascend310P'):
        pytest.skip('small-width vendor issue was reproduced on 310P')
    from jittor.backends.acl.kernels.ops.norms_op import RmsNormACL
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1), torch.no_grad(), forbid_backend_fallbacks():
        source, weight = _values(17, width)
        value, gamma = jt.array(source), jt.array(weight)
        with pytest.raises(RuntimeError, match='RmsNorm on Ascend310P rejects float16 normalized widths below 16'):
            result = (jt.core.ops.mapped_rms_norm(value, gamma, 1e-6) if route == 'mapped'
                      else RmsNormACL()(value, gamma, 1e-6))
            result.sync(True)
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize('boundary', ['float32', 'weight_float32', 'multi_axis', 'weight_none',
                                     'strided_input', 'strided_weight', 'zero_epsilon'])
def test_stock_rms_norm_unverified_boundaries_keep_portable_device_path(boundary, monkeypatch):
    _require_acl()
    calls = _record_dispatch(monkeypatch)
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, auto_graph_replay=0, transpose_storage_view=1), torch.no_grad(), forbid_backend_fallbacks():
        source, weight = _values(3, 32)
        axes = (-1,)
        dtype = torch.float32 if boundary == 'float32' else torch.float16
        source = source.astype('float32') if boundary == 'float32' else source
        module = _module(weight.astype('float32') if boundary == 'float32' else weight, dtype=dtype)
        value = torch.tensor(source, dtype=dtype, device='cuda')
        if boundary == 'weight_float32':
            module.weight = torch.nn.Parameter(torch.tensor(weight.astype('float32'), device='cuda'))
        elif boundary == 'multi_axis':
            weight = np.ones(source.shape, dtype='float16') * np.float16(.75)
            module = _module(weight)
            axes = (-2, -1)
        elif boundary == 'weight_none':
            module.weight = None
            weight = np.ones((32,), dtype='float16')
        elif boundary == 'strided_input':
            value = torch.tensor(source.T.copy(), dtype=dtype, device='cuda').transpose(0, 1)
            assert not value._storage_is_contiguous()
        elif boundary == 'strided_weight':
            values = np.stack([weight, weight + np.float16(.2)], axis=1)
            module.weight = torch.nn.Parameter(torch.tensor(values, dtype=dtype, device='cuda')[:, 0])
            assert not module.weight._storage_is_contiguous()
        elif boundary == 'zero_epsilon':
            module.eps = 0.
        result = module(value)
        jt.sync_all(True)
        assert result.location() == 'device' and result.dtype == dtype
        np.testing.assert_allclose(result.cpu().numpy(), _reference(source, weight, module.eps, axes),
                                   rtol=3e-3 if dtype == torch.float16 else 2e-5,
                                   atol=3e-3 if dtype == torch.float16 else 2e-5)
    assert calls == []
    assert jt.core.backend_fallback_count() == before


def test_stock_rms_norm_training_keeps_input_and_weight_gradients(monkeypatch):
    _require_acl()
    calls = _record_dispatch(monkeypatch)
    source, weight = _values(3, 32)
    cotangent = np.random.default_rng(2).normal(0, .2, source.shape).astype('float16')
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, auto_graph_replay=0), torch.enable_grad(), forbid_backend_fallbacks():
        value = torch.tensor(source, dtype=torch.float16, device='cuda', requires_grad=True)
        module = _module(weight)
        result = module(value)
        dx, dw = torch.autograd.grad((result * torch.tensor(cotangent, device='cuda')).sum(),
                                     (value, module.weight))
        jt.sync_all(True)
        x, w, dy = (item.astype(np.float64) for item in (source, weight, cotangent))
        inverse = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + module.eps)
        scaled = dy * w
        expected_dx = scaled * inverse - x * (scaled * x).mean(-1, keepdims=True) * inverse ** 3
        expected_dw = (dy * x * inverse).sum(0)
        assert all(item.location() == 'device' for item in (result, dx, dw))
        assert np.isfinite(dx.cpu().numpy()).all() and np.isfinite(dw.cpu().numpy()).all()
        np.testing.assert_allclose(dx.cpu().numpy(), expected_dx, rtol=6e-3, atol=2e-3)
        np.testing.assert_allclose(dw.cpu().numpy(), expected_dw, rtol=6e-3, atol=2e-3)
    assert calls == []
    assert jt.core.backend_fallback_count() == before
