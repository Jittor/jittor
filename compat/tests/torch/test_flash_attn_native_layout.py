"""Native dense FlashAttention must preserve logical noncontiguous views."""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get('JITTOR_FLASH_ATTN_JITTOR_SRC'),
    reason='requires official FlashAttention CUDA backend',
)


@pytest.mark.parametrize('layout', ['contiguous', 'transposed'])
def test_native_dense_gqa_inference_respects_view_layout(layout):
    import jittor as jt
    import torch
    import flash_attn
    from jittor.compat.shim.backends import flash_attention

    rng = np.random.default_rng(112)
    raw = [rng.normal(size=shape).astype(np.float32) for shape in
           ((1, 32, 4, 32), (1, 32, 2, 32), (1, 32, 2, 32))]
    with torch.no_grad(), jt.flag_scope(use_cuda=1):
        q, k, v = [torch.tensor(a, device='cuda').to(torch.bfloat16) for a in raw]
        if layout == 'transposed':
            q = q.transpose(1, 2).contiguous().transpose(1, 2)
            k = k.transpose(1, 2).contiguous().transpose(1, 2)
            assert not q.is_contiguous() and not k.is_contiguous()
        source = [t.detach().clone().cpu().float().numpy() for t in (q, k, v)]
        output = flash_attn.flash_attn_func(q, k, v, causal=True)
        assert flash_attn.flashattn_jittor_backend().startswith('flashattn_jittor_official:')
        assert flash_attention.load_backend()._flashattn_jittor_packed_low_level is not None
        actual = output.detach().clone().cpu().float().numpy()

    queries, keys, values = source
    expected = np.empty_like(queries)
    causal_mask = np.triu(np.ones((32, 32), dtype=bool), 1)
    for h in range(4):
        score = queries[0, :, h] @ keys[0, :, h // 2].T / np.sqrt(32)
        score[causal_mask] = -np.inf
        score -= score.max(axis=-1, keepdims=True)
        probability = np.exp(score); probability /= probability.sum(axis=-1, keepdims=True)
        expected[0, :, h] = probability @ values[0, :, h // 2]
    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.02)
