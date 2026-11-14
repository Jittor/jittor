"""Real 910B acceptance of padded noncausal FP16 self attention."""
import math

import jittor as jt
import numpy as np
import pytest

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels._soc import soc_name
from jittor.backends.acl.kernels.ops import flashattention_op as attention


@pytest.fixture(autouse=True)
def real_910b():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, transpose_storage_view=1), jt.no_grad(), forbid_backend_fallbacks():
        if not soc_name().startswith("Ascend910B"):
            pytest.skip("requires Ascend910B")
        yield
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("batch,heads,length,width,strided,scale", [
    (1, 2, 128, 64, False, None),
    (2, 4, 129, 128, True, .07),
    (1, 32, 1797, 64, True, None),
    (1, 2, 257, 16, False, None),
    (1, 2, 129, 256, False, None),
])
def test_public_sdpa_dispatch_and_numerics(batch, heads, length, width, strided, scale, monkeypatch):
    rng = np.random.default_rng(150)
    hosts = [rng.normal(0, .7, (batch, heads, length, width)).astype(np.float16)
             for _ in range(3)]
    if strided:
        tensors = [jt.core.ops.transpose(jt.array(x.transpose(0, 2, 1, 3).copy()), (0, 2, 1, 3))
                   for x in hosts]
        assert all(not x._storage_is_contiguous() for x in tensors)
    else:
        tensors = [jt.array(x) for x in hosts]
    calls = []
    original = attention.PromptFlashAttentionACL.__call__

    def record(self, q, k, v, drop_mask=None):
        calls.append((self.is_causal, tuple(q.shape), drop_mask is not None))
        return original(self, q, k, v, drop_mask)

    monkeypatch.setattr(attention.PromptFlashAttentionACL, "__call__", record)
    padded = (length + 127) // 128 * 128
    for _ in range(2):
        result = jt.nn.scaled_dot_product_attention(*tensors, scale=scale)
        result.sync(True)
        assert result.location() == "device" and str(result.dtype) == "float16"
        actual = result.numpy()
        # Head-wise FP64 oracle bounds host memory for long attention.
        for b in range(batch):
            for h in range(heads):
                q, k, v = (x[b, h].astype(np.float64) for x in hosts)
                scores = (q @ k.T) * (scale if scale is not None else 1 / math.sqrt(width))
                weights = np.exp(scores - scores.max(-1, keepdims=True))
                expected = (weights / weights.sum(-1, keepdims=True)) @ v
                np.testing.assert_allclose(actual[b, h], expected, rtol=3e-3, atol=1e-3)
        for tensor, host in zip(tensors, hosts):
            copy = jt.Var.copy(tensor).sync(True)
            np.testing.assert_array_equal(copy.numpy(), host)
            assert tensor.location() == "device"
    assert calls == [(False, (batch, heads, padded, width), True)] * 2


def test_padding_does_not_dilute_constant_values():
    q = jt.zeros((1, 2, 129, 64), dtype="float16")
    result = jt.nn.scaled_dot_product_attention(q, q, jt.full(q.shape, .5, dtype="float16"))
    result.sync(True)
    assert result.location() == "device"
    np.testing.assert_allclose(result.numpy(), .5, rtol=0, atol=1e-3)
