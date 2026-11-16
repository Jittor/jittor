"""Requested SDPA backends change execution and restore state on scope exit."""
import os
from unittest import mock

import numpy as np
import pytest
import jittor as jt
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from jittor.compat.torch.installers.nn import attention
from _helpers import capability


def flags():
    cuda = torch.backends.cuda
    return tuple(getattr(cuda, name+'_sdp_enabled')() for name in
                 ('flash', 'mem_efficient', 'math', 'cudnn'))


def test_nested_selection_restores_flags_after_exception():
    before = flags()
    with sdpa_kernel(SDPBackend.MATH):
        assert flags() == (False, False, True, False)
        with pytest.raises(ValueError, match='sentinel'):
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                assert flags() == (False, True, False, False)
                raise ValueError('sentinel')
        assert flags() == (False, False, True, False)
    assert flags() == before


def test_math_policy_bypasses_fused_routes_and_preserves_gradients():
    rng = np.random.default_rng(782)
    arrays = [rng.normal(size=(1, 2, 5, 8)).astype('float32') for _ in range(3)]
    q, k, v = [torch.tensor(value, device='cpu', requires_grad=True) for value in arrays]
    score = arrays[0].astype('float64') @ arrays[1].astype('float64').swapaxes(-1, -2)/np.sqrt(8)
    probability = np.exp(score-score.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    expected = probability @ arrays[2]
    with sdpa_kernel(SDPBackend.MATH), mock.patch.object(
            attention, '_try_flash_scaled_dot_product_attention', side_effect=AssertionError('Flash used')):
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        output.sum().backward()
    np.testing.assert_allclose(output.detach().numpy(), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(v.grad.numpy(), np.broadcast_to(
        probability.sum(-2)[..., None], v.shape), rtol=1e-5, atol=1e-6)


def test_empty_selection_cannot_silently_use_math():
    value = torch.ones((1, 2, 3, 8), device='cpu')
    with sdpa_kernel([]), pytest.raises(RuntimeError, match='No available kernel'):
        torch.nn.functional.scaled_dot_product_attention(value, value, value)


def test_legacy_scope_uses_and_restores_the_same_policy():
    before = flags()
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False,
                                        enable_mem_efficient=True, enable_cudnn=False):
        assert flags() == (False, True, False, False)
    assert flags() == before


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_forced_efficient_backend_executes_real_cuda_forward_and_backward(dtype):
    if not capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip('CUDA is unavailable')
    if not os.environ.get('JITTOR_EFFICIENT_ATTN_SRC'):
        pytest.skip('upstream CUTLASS attention source is not configured')
    from jittor.backends.cuda.kernels.nn import efficient_attention_cuda as provider
    rng = np.random.default_rng(812)
    with jt.flag_scope(use_cuda=1, backend_fallback='error'):
        values = [torch.tensor(rng.normal(0, .2, (1, 2, 33, 64)).astype('float32'),
                               device='cuda', dtype=dtype, requires_grad=True) for _ in range(3)]
        before = jt.core.backend_fallback_count()
        calls = []
        original = provider._library

        def library(*args):
            calls.append(args)
            return original(*args)

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), mock.patch.object(provider, '_library', library):
            output = torch.nn.functional.scaled_dot_product_attention(*values)
            output.float().square().mean().backward()
            output.sync()
        assert calls and output.dtype == dtype and output.device.type == 'cuda'
        for value in values:
            assert value.grad is not None and value.grad.device.type == 'cuda'
            assert np.isfinite(value.grad.float().cpu().numpy()).all()
        assert jt.core.backend_fallback_count() == before


def test_cuda_autocast_converts_float_bias_with_attention_inputs():
    if not capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip('CUDA is unavailable')
    if not os.environ.get('JITTOR_EFFICIENT_ATTN_SRC'):
        pytest.skip('upstream CUTLASS attention source is not configured')
    with jt.flag_scope(use_cuda=1, backend_fallback='error'):
        values = [torch.randn(1, 2, 17, 64, device='cuda', requires_grad=True) for _ in range(3)]
        mask = torch.zeros((1, 1, 17, 17), device='cuda')
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), torch.autocast('cuda', dtype=torch.bfloat16):
            output = torch.nn.functional.scaled_dot_product_attention(*values, attn_mask=mask)
            assert output.dtype == torch.bfloat16
        output.float().sum().backward()
        for value in values:
            assert value.grad.dtype == torch.float32 and value.grad.device.type == 'cuda'
            assert np.isfinite(value.grad.detach().cpu().numpy()).all()


@pytest.mark.parametrize('query_length,key_length', [(17, 17), (17, 29), (29, 17)])
def test_flash_only_short_training_does_not_choose_disabled_math(monkeypatch, query_length, key_length):
    if not capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip('CUDA is unavailable')
    if not os.environ.get('JITTOR_FLASH_ATTN_JITTOR_SRC'):
        pytest.skip('requires official FlashAttention CUDA backend')
    from jittor.compat.diagnostics import sdpa_flash_stats

    monkeypatch.setenv('JITTOR_FLASH_ATTN_JITTOR_REQUIRED', '0')
    monkeypatch.setenv('JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES', str(1 << 24))
    rng = np.random.default_rng(714)
    shapes = [(1, 2, query_length, 64), (1, 2, key_length, 64),
              (1, 2, key_length, 64)]
    values = [torch.tensor(rng.normal(0, .2, shape).astype('float32'), device='cuda',
                           dtype=torch.bfloat16, requires_grad=True) for shape in shapes]
    q, k, v = [x.detach().float().cpu().numpy().astype('float64') for x in values]
    dy = torch.tensor(rng.normal(size=shapes[0]).astype('float32'), device='cuda',
                      dtype=torch.bfloat16)
    cotangent = dy.float().cpu().numpy().astype('float64')
    scores = q @ k.swapaxes(-1, -2) * .125
    scores[..., np.triu(np.ones((query_length, key_length), bool), 1)] = -np.inf
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    dp = cotangent @ v.swapaxes(-1, -2)
    ds = p * (dp - (p * dp).sum(-1, keepdims=True))
    expected = [p @ v, ds @ k * .125, ds.swapaxes(-1, -2) @ q * .125,
                p.swapaxes(-1, -2) @ cotangent]
    hits = sdpa_flash_stats(jt)['hits']
    fallback = jt.core.backend_fallback_count()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        result = torch.nn.functional.scaled_dot_product_attention(*values, is_causal=True)
        (result.float() * dy.float()).sum().backward()
    assert sdpa_flash_stats(jt)['hits'] > hits
    assert jt.core.backend_fallback_count() == fallback
    for actual, wanted in zip([result] + [x.grad for x in values], expected):
        assert actual.device.type == 'cuda'
        np.testing.assert_allclose(actual.detach().float().cpu().numpy(), wanted,
                                   rtol=.04, atol=max(1e-7, np.abs(wanted).max()*.004))
