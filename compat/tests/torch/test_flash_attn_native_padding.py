"""Oracle for native packed-varlen inference with unaligned head dimensions."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"),
    reason="requires the official FlashAttention CUDA source backend",
)


@pytest.mark.parametrize("head_dim", [18, 24])
@pytest.mark.parametrize("explicit_scale", [False, True])
def test_native_varlen_qkv_packed_padding(head_dim, explicit_scale):
    import jittor as jt
    import torch
    import flash_attn
    from jittor.compat.shim.backends import flash_attention

    rng = np.random.default_rng(17)
    raw = rng.standard_normal((5, 3, 2, head_dim)).astype(np.float32)
    scale = 0.25 if explicit_scale else None
    with torch.no_grad(), jt.flag_scope(use_cuda=1):
        qkv = torch.tensor(raw, device="cuda").to(torch.bfloat16)
        assert not qkv.requires_grad
        cu = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
        rounded = qkv.float().numpy()
        result = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv, cu, 3, softmax_scale=scale, dropout_p=0.0
        )
        assert result.is_cuda
        assert result.shape == (5, 2, head_dim)
        assert flash_attn.flashattn_jittor_backend().startswith("flashattn_jittor_official:")
        assert flash_attention.load_backend()._flashattn_jittor_packed_low_level is not None
        actual = result.float().numpy()

    expected = []
    factor = head_dim ** -0.5 if scale is None else scale
    for start, end in ((0, 3), (3, 5)):
        q, k, v = (rounded[start:end, i] for i in range(3))
        scores = np.einsum("lhd,shd->hls", q, k) * factor
        scores -= scores.max(axis=-1, keepdims=True)
        prob = np.exp(scores)
        prob /= prob.sum(axis=-1, keepdims=True)
        expected.append(np.einsum("hls,shd->lhd", prob, v))
    np.testing.assert_allclose(actual, np.concatenate(expected), atol=0.02, rtol=0.02)
