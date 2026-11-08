"""Native packed variable-length training with padded attention heads."""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get('JITTOR_FLASH_ATTN_JITTOR_SRC'),
    reason='requires the official FlashAttention CUDA source backend',
)


@pytest.mark.parametrize('head_dim', [18, 24])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('explicit_scale', [False, True])
def test_varlen_packed_training_output_and_grad(head_dim, causal, explicit_scale):
    import jittor as jt
    import torch
    import flash_attn

    rng = np.random.default_rng(900 + head_dim)
    raw = rng.standard_normal((5, 3, 2, head_dim)).astype(np.float32)
    weights = rng.standard_normal((5, 2, head_dim)).astype(np.float32)
    factor = .25 if explicit_scale else head_dim ** -.5
    with jt.flag_scope(use_cuda=1):
        qkv = torch.tensor(raw, device='cuda').to(torch.bfloat16).requires_grad_(True)
        cu = torch.tensor([0, 3, 5], device='cuda', dtype=torch.int32)
        grad_seed = torch.tensor(weights, device='cuda')
        output = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv, cu, 3, causal=causal, dropout_p=0.,
            softmax_scale=.25 if explicit_scale else None)
        loss = (output.float() * grad_seed).sum()
        grad = torch.autograd.grad(loss, qkv)[0]
        assert flash_attn.flashattn_jittor_backend().startswith('flashattn_jittor_official:')
        values = qkv.detach().float().cpu().numpy()
        seed = grad_seed.to(torch.bfloat16).float().cpu().numpy()
        actual_out = output.detach().float().cpu().numpy()
        actual_grad = grad.detach().float().cpu().numpy()

    outputs, gradients = [], []
    for begin, end in ((0, 3), (3, 5)):
        q, k, v = (values[begin:end, i] for i in range(3))
        dy = seed[begin:end]
        scores = np.einsum('lhd,shd->hls', q, k) * factor
        if causal:
            mask = np.triu(np.ones((end - begin, end - begin), dtype=bool), k=1)
            scores[:, mask] = -np.inf
        scores -= scores.max(axis=-1, keepdims=True)
        prob = np.exp(scores)
        prob /= prob.sum(axis=-1, keepdims=True)
        outputs.append(np.einsum('hls,shd->lhd', prob, v))
        dp = np.einsum('lhd,shd->hls', dy, v)
        ds = prob * (dp - (prob * dp).sum(axis=-1, keepdims=True))
        dq = factor * np.einsum('hls,shd->lhd', ds, k)
        dk = factor * np.einsum('hls,lhd->shd', ds, q)
        dv = np.einsum('hls,lhd->shd', prob, dy)
        gradients.append(np.stack((dq, dk, dv), axis=1))
    np.testing.assert_allclose(actual_out, np.concatenate(outputs), atol=.025, rtol=.025)
    np.testing.assert_allclose(actual_grad, np.concatenate(gradients), atol=.025, rtol=.025)
