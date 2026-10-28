"""310P FP16 causal prefill through public SDPA and CANN PFA V3."""

import math

import numpy as np
import pytest
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels._soc import soc_name
from jittor.backends.acl.kernels.ops import flashattention_op as attention


@pytest.fixture(autouse=True)
def acl_310p_inference():
    if not _test_capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, transpose_storage_view=1), jt.no_grad(), forbid_backend_fallbacks():
        if not soc_name().startswith("Ascend310P"):
            pytest.skip("PFA prefill selection is currently verified for Ascend310P")
        yield
    assert jt.core.backend_fallback_count() == before


def _reference(query, key, value, scale):
    group = query.shape[1] // key.shape[1]
    query = query.astype(np.float64)
    key = np.repeat(key.astype(np.float64), group, axis=1)
    value = np.repeat(value.astype(np.float64), group, axis=1)
    scores = query @ key.swapaxes(-1, -2) * scale
    blocked = np.triu(np.ones(scores.shape[-2:], dtype=bool), 1)
    scores = np.where(blocked, -np.inf, scores)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities @ value


def _inputs(batch, heads, kv_heads, length, width, strided=False, seed=107):
    rng = np.random.RandomState(seed)
    host = [rng.uniform(-0.3, 0.3, (batch, h, length, width)).astype(np.float16)
            for h in (heads, kv_heads, kv_heads)]
    if strided:
        tensors = [jt.core.ops.transpose(jt.array(value.transpose(0, 2, 1, 3).copy()), (0, 2, 1, 3))
                   for value in host]
        assert not tensors[0]._storage_is_contiguous()
    else:
        tensors = [jt.array(value) for value in host]
    return host, tensors


@pytest.mark.parametrize("batch,heads,kv_heads,length,width,strided", [
    (1, 2, 2, 17, 64, False),
    (1, 4, 2, 128, 64, True),
    (2, 4, 2, 129, 64, False),
    (1, 2, 2, 257, 128, True),
    (1, 4, 2, 502, 128, False),
    (1, 2, 1, 1998, 64, False),
])
def test_public_sdpa_causal_prefill_matches_independent_reference(
    batch, heads, kv_heads, length, width, strided, monkeypatch,
):
    host, tensors = _inputs(batch, heads, kv_heads, length, width, strided)
    scale = 1.0 / math.sqrt(width)
    expected = _reference(*host, scale)
    calls = []
    original_call = attention.PromptFlashAttentionACL.__call__

    def record(self, query, key, value, mask):
        calls.append((tuple(query.shape), tuple(key.shape), tuple(mask.shape)))
        return original_call(self, query, key, value, mask)

    monkeypatch.setattr(attention.PromptFlashAttentionACL, "__call__", record)
    output = jt.nn.scaled_dot_product_attention(
        *tensors, is_causal=True, enable_gqa=heads != kv_heads, scale=scale)
    outputs = [output]
    if heads != kv_heads:
        # The native public frontend expands GQA before dispatch, while the
        # Torch frontend dispatches compact KV heads. Exercise that backend
        # entry as well, against the same independent mathematical reference.
        outputs.append(attention.scaled_dot_product_attention_acl(
            *tensors, is_causal=True, enable_gqa=True, scale=scale))
    assert all(value is not None for value in outputs)
    jt.sync(outputs)
    padded = (length + 127) // 128 * 128
    assert len(calls) == len(outputs)
    assert all(q == k == (batch, heads, padded, width) and mask == (padded, padded)
               for q, k, mask in calls)
    assert attention.scaled_dot_product_attention_acl.backend_name == "acl_prompt_flash_attention_v3"
    for value in outputs:
        assert tuple(value.shape) == host[0].shape
        assert str(value.dtype) == "float16"
        assert value.location() == "device"
    for actual in jt.fetch_sync(outputs):
        assert np.isfinite(actual).all()
        # Same bound used by the standalone real-device probe: this validates
        # all output elements here, including the 502/1998-token cases.
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=4e-4)


def test_prompt_mask_cache_restores_device_residency_after_host_read():
    length, width = 17, 64
    host, tensors = _inputs(1, 2, 2, length, width)
    first = jt.nn.scaled_dot_product_attention(*tensors, is_causal=True)
    first.sync(True)
    device = int(jt.core.dispatch_context([tensors[0]])[1])
    cache_key = (device, length, 128)
    mask = attention._prompt_causal_mask_cache[cache_key]
    actual_mask = mask.numpy()
    assert mask.location() == "cpu"
    expected_mask = np.triu(np.ones((128, 128), dtype=bool), 1)
    expected_mask[:, length:] = True
    np.testing.assert_array_equal(actual_mask, expected_mask)
    assert not actual_mask.all(axis=-1).any()

    second_host, second_inputs = _inputs(1, 2, 2, length, width, seed=109)
    second = jt.nn.scaled_dot_product_attention(*second_inputs, is_causal=True)
    second.sync(True)
    assert second.location() == "device"
    restored = attention._prompt_causal_mask_cache[cache_key]
    assert restored.location() == "device"
    assert restored.device_id == device
    np.testing.assert_allclose(
        second.numpy(), _reference(*second_host, 1.0 / math.sqrt(width)), rtol=3e-3, atol=4e-4)
