"""Real 310P unpadded noncausal PFA for original rectangular Q/K lengths."""
import math

import numpy as np
import pytest
import jittor as jt

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels._soc import soc_name
from jittor.backends.acl.kernels.ops import flashattention_op as attention


@pytest.fixture(autouse=True)
def real_310p():
    if not capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip('real ACL device required')
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, transpose_storage_view=1), jt.no_grad(), forbid_backend_fallbacks():
        if not soc_name().startswith('Ascend310P'):
            pytest.skip('unpadded PFA is verified on 310P')
        yield
    assert jt.core.backend_fallback_count() == before


def _inputs(batch, heads, query, source, width, strided=False, seed=107):
    rng = np.random.default_rng(seed)
    hosts = [rng.normal(0, .7, (batch, heads, length, width)).astype(np.float16)
             for length in (query, source, source)]
    if strided:
        values = [jt.core.ops.transpose(jt.array(host.transpose(0, 2, 1, 3).copy()), (0, 2, 1, 3))
                  for host in hosts]
        assert all(not value._storage_is_contiguous() for value in values)
    else:
        values = [jt.array(host) for host in hosts]
    return hosts, values


def _reference(query, key, value, causal=False):
    q, k, v = (array.astype(np.float64) for array in (query, key, value))
    scores = q @ k.swapaxes(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        scores = np.where(np.triu(np.ones(scores.shape[-2:], bool), 1), -np.inf, scores)
    probabilities = np.exp(scores - scores.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    return probabilities @ v


def _record(monkeypatch):
    calls = []
    original = attention.PromptFlashAttentionACL.__call__
    def record(self, query, key, value, drop_mask=None):
        calls.append((self.is_causal, tuple(query.shape), tuple(key.shape), drop_mask))
        return original(self, query, key, value, drop_mask)
    monkeypatch.setattr(attention.PromptFlashAttentionACL, '__call__', record)
    return calls


@pytest.mark.parametrize('batch,heads,query,source,width,strided', [
    (1, 12, 196, 77, 64, False),
    (1, 10, 1024, 77, 64, True),
    (2, 10, 1024, 77, 64, False),
    (1, 20, 4096, 77, 64, False),
    (2, 20, 4096, 77, 64, True),
    (1, 24, 800, 512, 128, True),
    (1, 2, 19, 7, 64, False),
])
def test_rectangular_public_sdpa_uses_original_lengths_and_no_mask(
        batch, heads, query, source, width, strided, monkeypatch):
    hosts, values = _inputs(batch, heads, query, source, width, strided)
    calls = _record(monkeypatch)
    result = jt.nn.scaled_dot_product_attention(*values, is_causal=False)
    result.sync(True)
    assert calls == [(False, hosts[0].shape, hosts[1].shape, None)]
    assert result.location() == 'device' and str(result.dtype) == 'float16'
    assert tuple(result.shape) == hosts[0].shape
    actual = result.numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, _reference(*hosts), rtol=3e-3, atol=1e-3)


def test_self_attention_196_does_not_use_padding_or_mask_helpers(monkeypatch):
    hosts, values = _inputs(1, 12, 196, 196, 64)
    def unexpected(*args, **kwargs):
        pytest.fail('unmasked noncausal PFA must retain original tokens')
    monkeypatch.setattr(attention, '_prompt_mask', unexpected)
    monkeypatch.setattr(attention, '_prompt_pad_sequence', unexpected)
    result = jt.nn.scaled_dot_product_attention(*values, is_causal=False)
    result.sync(True)
    np.testing.assert_allclose(result.numpy(), _reference(*hosts), rtol=3e-3, atol=1e-3)


def test_constant_values_are_not_diluted_by_source_padding():
    _, values = _inputs(2, 10, 1024, 77, 64)
    constant = jt.full(values[2].shape, .5, dtype='float16')
    output = jt.nn.scaled_dot_product_attention(values[0], values[1], constant, is_causal=False)
    output.sync(True)
    assert output.location() == 'device'
    np.testing.assert_allclose(output.numpy(), .5, rtol=0, atol=1e-3)


def test_poisoned_storage_after_logical_source_does_not_enter_attention():
    hosts, values = _inputs(2, 10, 1024, 77, 64)
    padded_key = np.full((2, 10, 128, 64), 11, dtype=np.float16)
    padded_value = np.full((2, 10, 128, 64), -19, dtype=np.float16)
    padded_key[:, :, :77, :] = hosts[1]
    padded_value[:, :, :77, :] = hosts[2]
    key = jt.array(padded_key)[:, :, :77, :]
    value = jt.array(padded_value)[:, :, :77, :]
    assert not key._storage_is_contiguous() and not value._storage_is_contiguous()
    output = jt.nn.scaled_dot_product_attention(values[0], key, value, is_causal=False)
    output.sync(True)
    actual = output.numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, _reference(*hosts), rtol=3e-3, atol=1e-3)


def test_masked_and_maskless_programs_do_not_reuse_different_input_counts():
    for index, masked in enumerate((True, False, True, False)):
        hosts, values = _inputs(1, 2, 19, 19, 64, seed=300 + index)
        output = (attention._prompt_prefill_310p(*values, 64 ** -.5, is_causal=False)
                  if masked else jt.nn.scaled_dot_product_attention(*values, is_causal=False))
        output.sync(True)
        np.testing.assert_allclose(output.numpy(), _reference(*hosts), rtol=3e-3, atol=1e-3)
    assert (2, 64 ** -.5, False, 3) in attention._prompt_attention_programs
    assert (2, 64 ** -.5, False, 4) in attention._prompt_attention_programs


@pytest.mark.parametrize('boundary', ['causal_rectangular', 'explicit_mask', 'training'])
def test_rectangular_unverified_modes_keep_math_attention(boundary, monkeypatch):
    hosts, values = _inputs(1, 2, 19, 7, 64)
    calls = _record(monkeypatch)
    causal = boundary == 'causal_rectangular'
    mask = jt.ones((19, 7), dtype='bool') if boundary == 'explicit_mask' else None
    context = jt.enable_grad() if boundary == 'training' else jt.no_grad()
    with context:
        output = jt.nn.scaled_dot_product_attention(*values, is_causal=causal, attn_mask=mask)
        output.sync(True)
    assert calls == []
    assert output.location() == 'device'
    actual = output.numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, _reference(*hosts, causal=causal), rtol=4e-3, atol=2e-3)


def test_direct_causal_runner_requires_its_original_mask():
    _, values = _inputs(1, 2, 19, 19, 64)
    with pytest.raises(ValueError, match='Causal.*requires.*boolean drop-mask'):
        attention.PromptFlashAttentionACL(2, 64 ** -.5, is_causal=True)(*values)
