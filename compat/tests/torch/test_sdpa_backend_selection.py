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
